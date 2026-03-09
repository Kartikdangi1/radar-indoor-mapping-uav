#include "temporal_radar_mapping/temporal_radar_occupancy_grid.hpp"

namespace radar_mapping {

void IndoorRadarOccupancyGrid::decay_unobserved_cells_() {
  std::lock_guard<std::mutex> lock(grid_mutex_);

  // Map reset requested by odom_callback_ (bag loop detected)
  if (map_reset_requested_.exchange(false)) {
    RCLCPP_WARN(get_logger(), "Bag loop detected — clearing occupancy grid");
    rclcpp::Time init_time(0, 0, RCL_ROS_TIME);
    for (auto& cell : cells_) {
      cell = CellState{};
      cell.last_seen_time = init_time;
      cell.last_occupied_time = init_time;
    }
    std::fill(grid_msg_.data.begin(), grid_msg_.data.end(), -1);
    return;
  }

  rclcpp::Time now = this->now();

  for (size_t i = 0; i < cells_.size(); ++i) {
    CellState& cell = cells_[i];
    if (cell.last_seen_time.nanoseconds() == 0) continue;

    double age = (now - cell.last_seen_time).seconds();
    if (age <= params_.decay_delay_sec) continue;

    // Uniform decay for all cells (static/dynamic/uncertain)
    // To re-enable tiered decay (for studying dynamic obstacle behavior):
    //   1. Restore static_decay_factor and dynamic_decay_factor to Params struct (hpp)
    //   2. Uncomment the conditional below:
    // float decay;
    // if (cell.is_static_confirmed) {
    //   decay = static_cast<float>(params_.static_decay_factor);  // slower, walls persist
    // } else if (cell.is_dynamic) {
    //   decay = static_cast<float>(params_.dynamic_decay_factor);  // faster, clutter clears
    // } else {
    //   decay = static_cast<float>(params_.decay_factor);  // default, uncertain cells
    // }
    float decay = static_cast<float>(params_.decay_factor);

    if (cell.log_odds > 0) {
      cell.confidence *= decay;
      if (cell.confidence < 0.1f) {
        cell.log_odds *= decay;
      }
    } else {
      cell.log_odds *= decay;
      cell.confidence *= decay;
    }

    if (std::abs(cell.log_odds) < 0.01f && cell.confidence < 0.05f) {
      cell.log_odds = 0.0f;
      cell.confidence = 0.0f;
      cell.last_seen_time = rclcpp::Time(0, 0, RCL_ROS_TIME);
      cell.last_occupied_time = rclcpp::Time(0, 0, RCL_ROS_TIME);
      cell.observation_count = 0;
      cell.occupied_count = 0;
      cell.is_dynamic = false;
      cell.is_static_confirmed = false;
      cell.dynamic_observations = 0;
      cell.velocity_accumulator = 0.0f;
      cell.is_behind_detection = false;
    }
  }
}


void IndoorRadarOccupancyGrid::publish_filtered_cloud_(const std::vector<RadarDetection>& detections,
                                                       const rclcpp::Time& stamp) {
  sensor_msgs::msg::PointCloud2 cloud_msg;
  cloud_msg.header.stamp = stamp;
  cloud_msg.header.frame_id = params_.map_frame;

  size_t valid_count = 0;
  for (const auto& det : detections) {
    if (!det.is_occluded) valid_count++;
  }

  cloud_msg.fields.resize(5);

  cloud_msg.fields[0].name = "x";
  cloud_msg.fields[0].offset = 0;
  cloud_msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud_msg.fields[0].count = 1;

  cloud_msg.fields[1].name = "y";
  cloud_msg.fields[1].offset = 4;
  cloud_msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud_msg.fields[1].count = 1;

  cloud_msg.fields[2].name = "z";
  cloud_msg.fields[2].offset = 8;
  cloud_msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud_msg.fields[2].count = 1;

  cloud_msg.fields[3].name = "rcs";
  cloud_msg.fields[3].offset = 12;
  cloud_msg.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud_msg.fields[3].count = 1;

  cloud_msg.fields[4].name = "velocity";
  cloud_msg.fields[4].offset = 16;
  cloud_msg.fields[4].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud_msg.fields[4].count = 1;

  cloud_msg.point_step = 20;
  cloud_msg.row_step = cloud_msg.point_step * valid_count;
  cloud_msg.height = 1;
  cloud_msg.width = valid_count;
  cloud_msg.is_dense = true;
  cloud_msg.is_bigendian = false;
  cloud_msg.data.resize(cloud_msg.row_step);

  size_t idx = 0;
  for (const auto& det : detections) {
    if (det.is_occluded) continue;

    uint8_t* ptr = &cloud_msg.data[idx * cloud_msg.point_step];
    *reinterpret_cast<float*>(ptr + 0) = static_cast<float>(det.x_map);
    *reinterpret_cast<float*>(ptr + 4) = static_cast<float>(det.y_map);
    *reinterpret_cast<float*>(ptr + 8) = 0.0f;  // Set Z to zero for 2D visualization
    *reinterpret_cast<float*>(ptr + 12) = det.rcs_u8;
    *reinterpret_cast<float*>(ptr + 16) = det.velocity;
    idx++;
  }

  pub_filtered_cloud_->publish(cloud_msg);
}


void IndoorRadarOccupancyGrid::publish_grid_() {
  std::lock_guard<std::mutex> lock(grid_mutex_);
  grid_msg_.header.stamp = this->now();

  // Publish grid directly without Y-flip (flip is now applied at point cloud source)
  for (int grid_idx = 0; grid_idx < static_cast<int>(cells_.size()); ++grid_idx) {
    const CellState& cell = cells_[grid_idx];

    if (cell.confidence < params_.confidence_publish_threshold &&
        cell.observation_count < 3) {
      grid_msg_.data[grid_idx] = -1;
      continue;
    }

    if (std::abs(cell.log_odds) < 0.05f) {
      grid_msg_.data[grid_idx] = -1;
      continue;
    }

    float prob = probability_from_logodds_(cell.log_odds);

    // Isolated-cell suppression: occupied cells need at least N occupied neighbors
    if (prob > 0.5f && params_.min_occupied_neighbors > 0) {
      int gx = grid_idx % grid_width_;
      int gy = grid_idx / grid_width_;
      int n = 0;
      for (int dy = -1; dy <= 1 && n < params_.min_occupied_neighbors; ++dy) {
        for (int dx = -1; dx <= 1 && n < params_.min_occupied_neighbors; ++dx) {
          if (dx == 0 && dy == 0) continue;
          int nx = gx + dx, ny = gy + dy;
          if (nx >= 0 && nx < grid_width_ && ny >= 0 && ny < grid_height_) {
            if (cells_[grid_index_(nx, ny)].log_odds > 0.1f) ++n;
          }
        }
      }
      if (n < params_.min_occupied_neighbors) {
        grid_msg_.data[grid_idx] = -1;
        continue;
      }
    }

    int8_t occ = static_cast<int8_t>(std::clamp(
        static_cast<int>(std::round(prob * 100.0f)), 0, 100));
    grid_msg_.data[grid_idx] = occ;
  }

  pub_grid_->publish(grid_msg_);
  publish_scale_bar_();
}


void IndoorRadarOccupancyGrid::publish_scale_bar_() {
  if (!have_robot_pose_) return;

  visualization_msgs::msg::MarkerArray markers;
  auto stamp = this->now();
  int id = 0;

  // Scale bar: centered at 0, ticks from -3m to +3m
  const int ticks_per_side = 3;       // -3, -2, -1, 0, +1, +2, +3
  const double tick_interval = 1.0;
  const double bar_offset_y = -2.0;   // place bar 2m below the robot
  const double bar_z = 0.05;
  const double tick_height = 0.15;
  const double line_width = 0.06;

  // Bar center is at the robot's X position
  double bar_center_x = robot_x_in_map_;
  double bar_y = robot_y_in_map_ + bar_offset_y;
  double bar_left = bar_center_x - ticks_per_side * tick_interval;
  double bar_right = bar_center_x + ticks_per_side * tick_interval;

  // --- Main horizontal bar ---
  visualization_msgs::msg::Marker bar;
  bar.header.frame_id = params_.map_frame;
  bar.header.stamp = stamp;
  bar.ns = "scale_bar";
  bar.id = id++;
  bar.type = visualization_msgs::msg::Marker::LINE_STRIP;
  bar.action = visualization_msgs::msg::Marker::ADD;
  bar.scale.x = line_width;
  bar.color.r = 1.0f; bar.color.g = 1.0f; bar.color.b = 1.0f; bar.color.a = 1.0f;
  bar.lifetime = rclcpp::Duration::from_seconds(2.0);

  geometry_msgs::msg::Point p0, p1;
  p0.x = bar_left; p0.y = bar_y; p0.z = bar_z;
  p1.x = bar_right; p1.y = bar_y; p1.z = bar_z;
  bar.points.push_back(p0);
  bar.points.push_back(p1);
  markers.markers.push_back(bar);

  // --- Tick marks and labels from -3m to +3m ---
  for (int t = -ticks_per_side; t <= ticks_per_side; ++t) {
    double tx = bar_center_x + t * tick_interval;

    // Vertical tick (taller at 0)
    visualization_msgs::msg::Marker tick;
    tick.header.frame_id = params_.map_frame;
    tick.header.stamp = stamp;
    tick.ns = "scale_bar";
    tick.id = id++;
    tick.type = visualization_msgs::msg::Marker::LINE_STRIP;
    tick.action = visualization_msgs::msg::Marker::ADD;
    tick.scale.x = (t == 0) ? line_width * 1.5 : line_width;
    tick.color.r = 1.0f; tick.color.g = 1.0f; tick.color.b = 1.0f; tick.color.a = 1.0f;
    tick.lifetime = rclcpp::Duration::from_seconds(2.0);

    double th = (t == 0) ? tick_height * 1.5 : tick_height;
    geometry_msgs::msg::Point tb, tt;
    tb.x = tx; tb.y = bar_y - th / 2.0; tb.z = bar_z;
    tt.x = tx; tt.y = bar_y + th / 2.0; tt.z = bar_z;
    tick.points.push_back(tb);
    tick.points.push_back(tt);
    markers.markers.push_back(tick);

    // Text label
    visualization_msgs::msg::Marker label;
    label.header.frame_id = params_.map_frame;
    label.header.stamp = stamp;
    label.ns = "scale_bar";
    label.id = id++;
    label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    label.action = visualization_msgs::msg::Marker::ADD;
    label.pose.position.x = tx;
    label.pose.position.y = bar_y - th / 2.0 - 0.2;
    label.pose.position.z = bar_z;
    label.scale.z = 0.25;
    label.color.r = 1.0f; label.color.g = 1.0f; label.color.b = 1.0f; label.color.a = 1.0f;
    label.lifetime = rclcpp::Duration::from_seconds(2.0);

    if (t == 0) {
      label.text = "0";
    } else if (t > 0) {
      label.text = "+" + std::to_string(t) + "m";
    } else {
      label.text = std::to_string(t) + "m";
    }
    markers.markers.push_back(label);
  }

  // --- Robot trajectory ---
  if (trajectory_points_.size() >= 2) {
    visualization_msgs::msg::Marker traj;
    traj.header.frame_id = params_.map_frame;
    traj.header.stamp = stamp;
    traj.ns = "trajectory";
    traj.id = id++;
    traj.type = visualization_msgs::msg::Marker::LINE_STRIP;
    traj.action = visualization_msgs::msg::Marker::ADD;
    traj.scale.x = 0.05;  // line width
    traj.color.r = 0.0f; traj.color.g = 1.0f; traj.color.b = 0.3f; traj.color.a = 0.9f;
    traj.lifetime = rclcpp::Duration::from_seconds(0.0);  // persistent
    traj.points = trajectory_points_;
    markers.markers.push_back(traj);
  }

  pub_scale_bar_->publish(markers);
}

}  // namespace radar_mapping
