#include "temporal_radar_mapping/temporal_radar_occupancy_grid.hpp"

namespace radar_mapping {

void IndoorRadarOccupancyGrid::odom_callback_(const nav_msgs::msg::Odometry::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(odom_mutex_);

  OdomPose pose;
  pose.stamp = msg->header.stamp;
  pose.x = msg->pose.pose.position.x;
  pose.y = msg->pose.pose.position.y;
  pose.z = msg->pose.pose.position.z;
  pose.qx = msg->pose.pose.orientation.x;
  pose.qy = msg->pose.pose.orientation.y;
  pose.qz = msg->pose.pose.orientation.z;
  pose.qw = msg->pose.pose.orientation.w;
  pose.vx = msg->twist.twist.linear.x;
  pose.vy = msg->twist.twist.linear.y;
  pose.vz = msg->twist.twist.linear.z;

  // Detect bag loop: backward timestamp jump triggers optional grid reset
  if (params_.enable_map_reset_on_loop && have_fused_odom_) {
    double dt = (pose.stamp - latest_odom_.stamp).seconds();
    if (dt < -params_.map_reset_time_jump_sec) {
      map_reset_requested_.store(true);
      odom_buffer_.clear();
      trajectory_points_.clear();
    }
  }

  latest_odom_ = pose;
  have_fused_odom_ = true;

  // Update robot state from fused odometry (more accurate than TF differentiation)
  robot_x_in_map_ = pose.x;
  robot_y_in_map_ = pose.y;
  robot_vx_ = pose.vx;
  robot_vy_ = pose.vy;
  robot_yaw_ = std::atan2(
      2.0 * (pose.qw * pose.qz + pose.qx * pose.qy),
      1.0 - 2.0 * (pose.qy * pose.qy + pose.qz * pose.qz));
  have_robot_pose_ = true;
  have_robot_velocity_ = true;

  // Record trajectory point if robot moved enough
  double dx = pose.x - traj_last_x_;
  double dy = pose.y - traj_last_y_;
  if (trajectory_points_.empty() || (dx * dx + dy * dy) >= traj_min_dist_ * traj_min_dist_) {
    geometry_msgs::msg::Point pt;
    pt.x = pose.x;
    pt.y = pose.y;
    pt.z = 0.05;
    trajectory_points_.push_back(pt);
    traj_last_x_ = pose.x;
    traj_last_y_ = pose.y;
  }

  // Maintain circular buffer
  odom_buffer_.push_back(pose);
  while (static_cast<int>(odom_buffer_.size()) > params_.odom_buffer_size) {
    odom_buffer_.pop_front();
  }
}


bool IndoorRadarOccupancyGrid::interpolate_odom_(const rclcpp::Time& stamp, OdomPose& result) const {
  if (odom_buffer_.size() < 2) {
    if (!odom_buffer_.empty()) {
      result = odom_buffer_.back();
      return true;
    }
    return false;
  }

  // Find bracketing entries
  for (size_t i = 1; i < odom_buffer_.size(); ++i) {
    const auto& before = odom_buffer_[i - 1];
    const auto& after = odom_buffer_[i];

    if (stamp >= before.stamp && stamp <= after.stamp) {
      double dt_total = (after.stamp - before.stamp).seconds();
      if (dt_total < 1e-6) {
        result = before;
        return true;
      }
      double dt = (stamp - before.stamp).seconds();
      double alpha = dt / dt_total;

      result.stamp = stamp;
      result.x = before.x + alpha * (after.x - before.x);
      result.y = before.y + alpha * (after.y - before.y);
      result.z = before.z + alpha * (after.z - before.z);
      result.vx = before.vx + alpha * (after.vx - before.vx);
      result.vy = before.vy + alpha * (after.vy - before.vy);
      result.vz = before.vz + alpha * (after.vz - before.vz);
      // SLERP for quaternion
      double dot = before.qx * after.qx + before.qy * after.qy +
                   before.qz * after.qz + before.qw * after.qw;
      if (dot < 0) {
        // Negate to take shortest path
        result.qx = before.qx + alpha * (-after.qx - before.qx);
        result.qy = before.qy + alpha * (-after.qy - before.qy);
        result.qz = before.qz + alpha * (-after.qz - before.qz);
        result.qw = before.qw + alpha * (-after.qw - before.qw);
      } else {
        result.qx = before.qx + alpha * (after.qx - before.qx);
        result.qy = before.qy + alpha * (after.qy - before.qy);
        result.qz = before.qz + alpha * (after.qz - before.qz);
        result.qw = before.qw + alpha * (after.qw - before.qw);
      }
      // Normalize quaternion
      double norm = std::sqrt(result.qx * result.qx + result.qy * result.qy +
                              result.qz * result.qz + result.qw * result.qw);
      if (norm > 1e-6) {
        result.qx /= norm;
        result.qy /= norm;
        result.qz /= norm;
        result.qw /= norm;
      }
      return true;
    }
  }

  // If stamp is after all buffered data, use latest
  if (stamp > odom_buffer_.back().stamp) {
    result = odom_buffer_.back();
    return true;
  }
  // If stamp is before all buffered data, use earliest
  result = odom_buffer_.front();
  return true;
}


void IndoorRadarOccupancyGrid::odom_pose_to_transform_(const OdomPose& pose,
                                                       geometry_msgs::msg::TransformStamped& transform) const {
  transform.header.stamp = pose.stamp;
  transform.header.frame_id = params_.map_frame;
  transform.child_frame_id = params_.base_frame;
  transform.transform.translation.x = pose.x;
  transform.transform.translation.y = pose.y;
  transform.transform.translation.z = pose.z;
  transform.transform.rotation.x = pose.qx;
  transform.transform.rotation.y = pose.qy;
  transform.transform.rotation.z = pose.qz;
  transform.transform.rotation.w = pose.qw;
}


bool IndoorRadarOccupancyGrid::transform_point_via_odom_(double x_radar, double y_radar, double z_radar,
                                                          const OdomPose& odom_pose,
                                                          const geometry_msgs::msg::TransformStamped& radar_to_base,
                                                          double& x_map, double& y_map, double& z_map) const {
  // Step 1: radar_frame -> base_link (static transform)
  tf2::Transform tf_radar_to_base;
  tf2::fromMsg(radar_to_base.transform, tf_radar_to_base);
  tf2::Vector3 pt_radar(x_radar, y_radar, z_radar);
  tf2::Vector3 pt_base = tf_radar_to_base * pt_radar;

  // Step 2: base_link -> odom (from fused odometry)
  tf2::Quaternion q_odom(odom_pose.qx, odom_pose.qy, odom_pose.qz, odom_pose.qw);
  tf2::Vector3 t_odom(odom_pose.x, odom_pose.y, odom_pose.z);
  tf2::Transform tf_base_to_map(q_odom, t_odom);

  tf2::Vector3 pt_map = tf_base_to_map * pt_base;
  x_map = pt_map.x();
  y_map = pt_map.y();
  z_map = pt_map.z();
  return true;
}

}  // namespace radar_mapping
