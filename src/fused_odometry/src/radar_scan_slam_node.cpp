/**
 * radar_scan_slam_node.cpp
 *
 * Lightweight scan-matching SLAM node for radar-based drift correction.
 *
 * Accumulates RANSAC-filtered radar inlier clouds into submaps, registers
 * consecutive submaps with GICP, and publishes a map→odom TF correction.
 * The existing occupancy grid node then resolves map→odom→base_link→ARS_548
 * automatically, placing every point at drift-corrected coordinates.
 *
 * Subscriptions:
 *   /fused_odom/radar_inliers   — RANSAC inlier point cloud (sensor frame)
 *   /fused_odom/odometry  — Current odom-frame pose
 *
 * Publications:
 *   TF: map → odom         — Drift correction (~10 Hz)
 *   /fused_odom/slam_submap     — Debug: current reference submap (optional)
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/gicp.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Geometry>
#include <mutex>
#include <deque>
#include <cmath>

// ── Submap metadata for loop closure ─────────────────────────────────────────
struct SubmapData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    Eigen::Vector3d position;  // Odom frame position when submap created
    int submap_id;
    rclcpp::Time stamp;
};

class RadarScanSlamNode : public rclcpp::Node {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RadarScanSlamNode() : Node("radar_scan_slam_node")
    {
        // ── Parameters ──────────────────────────────────────────────────────
        map_frame_  = declare_parameter("map_frame",  std::string("map"));
        odom_frame_ = declare_parameter("odom_frame", std::string("odom"));

        accumulation_scans_    = declare_parameter("accumulation_scans", 8);
        keyframe_min_distance_ = declare_parameter("keyframe_min_distance", 0.3);
        voxel_leaf_size_       = declare_parameter("voxel_leaf_size", 0.15);
        min_submap_points_     = declare_parameter("min_submap_points", 50);

        max_correspondence_distance_ = declare_parameter("max_correspondence_distance", 2.0);
        max_iterations_              = declare_parameter("max_iterations", 50);
        transformation_epsilon_      = declare_parameter("transformation_epsilon", 1e-4);
        euclidean_fitness_epsilon_   = declare_parameter("euclidean_fitness_epsilon", 1e-5);
        max_fitness_score_           = declare_parameter("max_fitness_score", 1.5);
        gicp_correspondence_randomness_ = declare_parameter("gicp_correspondence_randomness", 5);

        tf_publish_hz_    = declare_parameter("tf_publish_hz", 10.0);
        publish_submap_   = declare_parameter("publish_submap", true);
        max_correction_m_ = declare_parameter("max_correction_m", 0.15);
        max_correction_rad_ = declare_parameter("max_correction_rad", 0.05);

        enable_scan_to_map_ = declare_parameter("enable_scan_to_map", true);
        global_map_max_submaps_ = declare_parameter("global_map_max_submaps", 20);
        min_map_points_ = declare_parameter("min_map_points", 100);

        // ── Loop closure ─────────────────────────────────────────────────────
        enable_loop_closure_ = declare_parameter("enable_loop_closure", true);
        loop_closure_min_distance_ = declare_parameter("loop_closure_min_distance", 5.0);
        loop_closure_max_submaps_ = declare_parameter("loop_closure_max_submaps", 100);
        loop_closure_fitness_threshold_ = declare_parameter("loop_closure_fitness_threshold", 0.65);
        loop_closure_max_correction_m_ = declare_parameter("loop_closure_max_correction_m", 0.5);

        // ── State ───────────────────────────────────────────────────────────
        T_map_odom_ = Eigen::Matrix4d::Identity();
        keyframe_odom_pose_ = Eigen::Matrix4d::Identity();
        accumulated_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
        reference_submap_.reset(new pcl::PointCloud<pcl::PointXYZ>());
        global_map_.reset(new pcl::PointCloud<pcl::PointXYZ>());
        scan_count_ = 0;
        has_reference_ = false;
        has_odom_ = false;
        last_tf_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
        next_submap_id_ = 0;
        loop_closure_count_ = 0;

        // ── TF ──────────────────────────────────────────────────────────────
        tf_buffer_   = std::make_shared<tf2_ros::Buffer>(get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // ── GICP setup ──────────────────────────────────────────────────────
        gicp_.setMaxCorrespondenceDistance(max_correspondence_distance_);
        gicp_.setMaximumIterations(max_iterations_);
        gicp_.setTransformationEpsilon(transformation_epsilon_);
        gicp_.setEuclideanFitnessEpsilon(euclidean_fitness_epsilon_);
        gicp_.setCorrespondenceRandomness(gicp_correspondence_randomness_);

        // Scan-to-map GICP (separate instance with same hyperparams)
        gicp_stm_.setMaxCorrespondenceDistance(max_correspondence_distance_);
        gicp_stm_.setMaximumIterations(max_iterations_);
        gicp_stm_.setTransformationEpsilon(transformation_epsilon_);
        gicp_stm_.setEuclideanFitnessEpsilon(euclidean_fitness_epsilon_);
        gicp_stm_.setCorrespondenceRandomness(gicp_correspondence_randomness_);

        // ── Publishers ──────────────────────────────────────────────────────
        if (publish_submap_) {
            submap_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
                "/fused_odom/slam_submap", 1);
        }
        if (enable_loop_closure_) {
            loop_closure_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
                "/fused_odom/loop_closures", 1);
        }

        // ── Subscriptions ───────────────────────────────────────────────────
        inlier_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            "/fused_odom/radar_inliers", rclcpp::SensorDataQoS(),
            std::bind(&RadarScanSlamNode::inlierCallback, this, std::placeholders::_1));

        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/fused_odom/odometry", rclcpp::SensorDataQoS(),
            std::bind(&RadarScanSlamNode::odomCallback, this, std::placeholders::_1));

        // ── Timer: broadcast TF (sim-time aware, NOT wall timer) ────────────
        // Use rclcpp::create_timer with node interfaces + clock so the timer
        // ticks in sim-time when use_sim_time:=true (wall_timer would mix
        // wall-clock firing with sim-time stamps, causing TF buffer clears).
        {
            auto cb = std::bind(&RadarScanSlamNode::broadcastTF, this);
            tf_timer_ = rclcpp::create_timer(
                get_node_base_interface(),
                get_node_timers_interface(),
                get_clock(),
                rclcpp::Duration::from_seconds(1.0 / tf_publish_hz_),
                std::move(cb));
        }

        RCLCPP_INFO(get_logger(),
            "RadarScanSLAM started — accumulate %d scans, keyframe dist %.2f m, "
            "voxel %.3f m, GICP corr dist %.1f m, max fitness %.2f, "
            "max correction %.3f m / %.3f rad",
            accumulation_scans_, keyframe_min_distance_, voxel_leaf_size_,
            max_correspondence_distance_, max_fitness_score_,
            max_correction_m_, max_correction_rad_);
    }

private:
    // ── Odometry callback ───────────────────────────────────────────────────
    void odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr& msg)
    {
        std::lock_guard<std::mutex> lock(odom_mutex_);
        current_odom_pose_ = odomMsgToMatrix(msg);
        last_odom_stamp_ = rclcpp::Time(msg->header.stamp);
        has_odom_ = true;
    }

    // ── Inlier cloud callback ───────────────────────────────────────────────
    void inlierCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg)
    {
        if (!has_odom_) return;

        // 1. Look up TF: odom → cloud.frame_id
        geometry_msgs::msg::TransformStamped tf_odom_sensor;
        try {
            tf_odom_sensor = tf_buffer_->lookupTransform(
                odom_frame_, msg->header.frame_id, msg->header.stamp,
                rclcpp::Duration::from_seconds(0.2));
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                "SLAM TF lookup failed: %s", ex.what());
            return;
        }

        // 2. Convert to PCL and transform into odom frame
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_sensor(
            new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud_sensor);
        if (cloud_sensor->empty()) return;

        Eigen::Matrix4f T_odom_sensor = tfToMatrix4f(tf_odom_sensor);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_odom(
            new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(*cloud_sensor, *cloud_odom, T_odom_sensor);

        // 3. Append to accumulation buffer
        {
            std::lock_guard<std::mutex> lock(cloud_mutex_);
            *accumulated_cloud_ += *cloud_odom;
            scan_count_++;
        }

        // Check keyframe triggers: BOTH scan count AND distance must be met
        Eigen::Matrix4d current_odom;
        {
            std::lock_guard<std::mutex> lock(odom_mutex_);
            current_odom = current_odom_pose_;
        }

        double distance_moved = (current_odom.block<3,1>(0,3) -
                                  keyframe_odom_pose_.block<3,1>(0,3)).norm();

        // Require minimum scans AND minimum distance (not OR)
        bool enough_scans = (scan_count_ >= accumulation_scans_);
        bool enough_distance = (distance_moved >= keyframe_min_distance_);
        if (!enough_scans || !enough_distance) return;

        // ── Process submap ──────────────────────────────────────────────────
        pcl::PointCloud<pcl::PointXYZ>::Ptr submap;
        {
            std::lock_guard<std::mutex> lock(cloud_mutex_);
            submap = accumulated_cloud_;
            accumulated_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
            scan_count_ = 0;
        }

        // 4. Voxel downsample
        pcl::PointCloud<pcl::PointXYZ>::Ptr submap_down(
            new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setInputCloud(submap);
        voxel.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
        voxel.filter(*submap_down);

        if (static_cast<int>(submap_down->size()) < min_submap_points_) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                "SLAM submap too sparse (%zu pts < %d), skipping",
                submap_down->size(), min_submap_points_);
            // Do NOT reset keyframe_odom_pose_ — keep accumulating from
            // the same reference point so next attempt has correct delta
            return;
        }

        // 5. First submap → store as reference
        if (!has_reference_) {
            reference_submap_ = submap_down;
            keyframe_odom_pose_ = current_odom;
            has_reference_ = true;
            RCLCPP_INFO(get_logger(), "SLAM: initial reference submap (%zu pts)",
                        submap_down->size());
            publishSubmap(submap_down, msg->header.stamp);
            return;
        }

        // 6. GICP align current_submap against reference_submap
        Eigen::Matrix4d T_odom_delta = keyframe_odom_pose_.inverse() * current_odom;
        Eigen::Matrix4f initial_guess = T_odom_delta.cast<float>();

        gicp_.setInputSource(submap_down);
        gicp_.setInputTarget(reference_submap_);
        pcl::PointCloud<pcl::PointXYZ> aligned;
        gicp_.align(aligned, initial_guess);

        if (!gicp_.hasConverged()) {
            // Reset reference to the current submap so the next keyframe compares
            // two consecutive submaps with a fresh odometry delta.  Without this,
            // T_odom_delta grows with each failed keyframe until GICP can never
            // recover (cascade failure producing 80+ seconds of non-convergence).
            RCLCPP_WARN(get_logger(),
                "SLAM GICP did not converge, resetting reference to current submap");
            reference_submap_ = submap_down;
            keyframe_odom_pose_ = current_odom;
            publishSubmap(submap_down, msg->header.stamp);
            return;
        }

        double fitness = gicp_.getFitnessScore();
        if (fitness > max_fitness_score_) {
            RCLCPP_WARN(get_logger(),
                "SLAM GICP fitness %.3f > threshold %.3f, resetting reference to current submap",
                fitness, max_fitness_score_);
            // Reset reference so the next keyframe has a fresh odometry delta.
            // Keeping the old reference would cause T_odom_delta to accumulate
            // across rejected frames until GICP can never recover (cascade failure).
            reference_submap_ = submap_down;
            keyframe_odom_pose_ = current_odom;
            publishSubmap(submap_down, msg->header.stamp);
            return;
        }

        // 7. Compute and clamp correction
        Eigen::Matrix4f T_gicp = gicp_.getFinalTransformation();
        Eigen::Matrix4d T_correction =
            T_gicp.cast<double>() * T_odom_delta.inverse();

        // Clamp translational correction to prevent large jumps
        double corr_dx = T_correction(0, 3);
        double corr_dy = T_correction(1, 3);
        double corr_dz = T_correction(2, 3);
        double corr_trans = std::sqrt(corr_dx*corr_dx + corr_dy*corr_dy + corr_dz*corr_dz);

        // Extract rotation angle from correction
        Eigen::Matrix3d R_corr = T_correction.block<3,3>(0,0);
        Eigen::AngleAxisd aa(R_corr);
        double corr_angle = std::abs(aa.angle());

        if (corr_trans > max_correction_m_ || corr_angle > max_correction_rad_) {
            RCLCPP_WARN(get_logger(),
                "SLAM s2s correction too large (%.3f m, %.3f rad), resetting reference "
                "(max %.3f m, %.3f rad)",
                corr_trans, corr_angle, max_correction_m_, max_correction_rad_);
            // A large s2s correction is almost certainly a false minimum on sparse radar.
            // Running scan-to-map from this (wrong) pose would also find a spurious
            // correction. Reset reference and return — same as the other failure paths.
            reference_submap_ = submap_down;
            keyframe_odom_pose_ = current_odom;
            publishSubmap(submap_down, msg->header.stamp);
            return;
        } else {
            std::lock_guard<std::mutex> lock(tf_mutex_);
            T_map_odom_ = T_correction * T_map_odom_;
        }

        // ── Stage 2: Scan-to-Map GICP (if enabled and global map is ready) ────
        if (enable_scan_to_map_ && global_map_->size() >= static_cast<size_t>(min_map_points_)) {
            // Transform current submap to map frame
            pcl::PointCloud<pcl::PointXYZ> submap_map_frame;
            Eigen::Matrix4f T_mo = T_map_odom_.cast<float>();
            pcl::transformPointCloud(*submap_down, submap_map_frame, T_mo);

            // GICP against global map
            gicp_stm_.setInputSource(submap_map_frame.makeShared());
            gicp_stm_.setInputTarget(global_map_);
            pcl::PointCloud<pcl::PointXYZ> aligned_stm;
            gicp_stm_.align(aligned_stm);

            if (gicp_stm_.hasConverged() && gicp_stm_.getFitnessScore() < max_fitness_score_) {
                Eigen::Matrix4d T_stm = gicp_stm_.getFinalTransformation().cast<double>();

                // Clamp translational and rotational corrections (same as scan-to-scan)
                double corr_stm_dx = T_stm(0, 3);
                double corr_stm_dy = T_stm(1, 3);
                double corr_stm_dz = T_stm(2, 3);
                double corr_stm_trans = std::sqrt(corr_stm_dx*corr_stm_dx + corr_stm_dy*corr_stm_dy + corr_stm_dz*corr_stm_dz);

                Eigen::Matrix3d R_stm = T_stm.block<3,3>(0,0);
                Eigen::AngleAxisd aa_stm(R_stm);
                double corr_stm_angle = std::abs(aa_stm.angle());

                if (corr_stm_trans > max_correction_m_ || corr_stm_angle > max_correction_rad_) {
                    RCLCPP_WARN(get_logger(),
                        "Scan-to-map correction too large (%.3f m, %.3f rad), rejecting",
                        corr_stm_trans, corr_stm_angle);
                } else {
                    {
                        std::lock_guard<std::mutex> lock(tf_mutex_);
                        T_map_odom_ = T_stm * T_map_odom_;
                    }
                    RCLCPP_INFO(get_logger(),
                        "Scan-to-map: fitness=%.4f, correction dx=%.3f dy=%.3f "
                        "(%.3f m, %.3f rad)",
                        gicp_stm_.getFitnessScore(),
                        T_stm(0, 3), T_stm(1, 3),
                        corr_stm_trans, corr_stm_angle);
                }
            }
        }

        // ── Loop closure detection ──────────────────────────────────────────
        if (enable_loop_closure_) {
            detectLoopClosure(submap_down, current_odom, msg->header.stamp);
        }

        // ── Update submap history and rebuild global map ──────────────────────
        // Re-transform submap to final map frame (after both stages) and add to history
        {
            std::lock_guard<std::mutex> lock(tf_mutex_);
            pcl::PointCloud<pcl::PointXYZ> final_map_frame;
            pcl::transformPointCloud(*submap_down, final_map_frame, T_map_odom_.cast<float>());
            submap_history_.push_back(final_map_frame);

            // Store submap metadata for loop closure
            SubmapData submap_meta;
            submap_meta.cloud = submap_down;
            submap_meta.position = current_odom.block<3,1>(0,3);
            submap_meta.submap_id = next_submap_id_++;
            submap_meta.stamp = rclcpp::Time(msg->header.stamp);
            submap_metadata_.push_back(submap_meta);

            if (static_cast<int>(submap_history_.size()) > global_map_max_submaps_) {
                submap_history_.pop_front();
                submap_metadata_.pop_front();
            }

            // Rebuild global_map_ from history
            global_map_->clear();
            for (const auto& s : submap_history_) *global_map_ += s;
            pcl::VoxelGrid<pcl::PointXYZ> vg;
            vg.setInputCloud(global_map_);
            vg.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
            vg.filter(*global_map_);
        }

        // Replace reference submap and update keyframe pose
        reference_submap_ = submap_down;
        keyframe_odom_pose_ = current_odom;

        RCLCPP_INFO(get_logger(),
            "SLAM keyframe: fitness=%.4f, pts=%zu, correction dx=%.3f dy=%.3f "
            "(%.3f m, %.3f rad)",
            fitness, submap_down->size(),
            T_correction(0, 3), T_correction(1, 3),
            corr_trans, corr_angle);

        // 8. Publish debug submap
        publishSubmap(submap_down, msg->header.stamp);
    }

    // ── Loop closure detection ──────────────────────────────────────────────
    void detectLoopClosure(const pcl::PointCloud<pcl::PointXYZ>::Ptr& current_submap,
                          const Eigen::Matrix4d& current_odom,
                          const rclcpp::Time& stamp)
    {
        if (submap_metadata_.size() < 2) return;

        Eigen::Vector3d current_pos = current_odom.block<3,1>(0,3);
        visualization_msgs::msg::MarkerArray loop_markers;

        // Iterate through all past submaps (except the most recent ones)
        int candidate_count = 0;
        for (int i = 0; i < static_cast<int>(submap_metadata_.size()) - 1; ++i) {
            const auto& past_meta = submap_metadata_[i];
            double distance = (current_pos - past_meta.position).norm();

            // Skip submaps that are too recent or too close in odometry space
            if (distance < loop_closure_min_distance_) continue;

            // Limit number of candidates checked per frame
            candidate_count++;
            if (candidate_count > loop_closure_max_submaps_) break;

            // Try GICP matching between current and past submap
            gicp_.setInputSource(current_submap);
            gicp_.setInputTarget(past_meta.cloud);

            // Initial guess: rough odometry delta
            Eigen::Matrix4d T_delta = Eigen::Matrix4d::Identity();
            T_delta(0,3) = current_pos(0) - past_meta.position(0);
            T_delta(1,3) = current_pos(1) - past_meta.position(1);
            T_delta(2,3) = current_pos(2) - past_meta.position(2);

            pcl::PointCloud<pcl::PointXYZ> aligned;
            gicp_.align(aligned, T_delta.cast<float>());

            if (!gicp_.hasConverged()) continue;

            double fitness = gicp_.getFitnessScore();
            Eigen::Matrix4f T_gicp_f = gicp_.getFinalTransformation();
            Eigen::Matrix4d T_gicp = T_gicp_f.cast<double>();

            // Reject bad matches (GICP fitness = mean sq distance, lower = better)
            if (fitness > loop_closure_fitness_threshold_) continue;

            // Extract translation
            double corr_dx = T_gicp(0, 3);
            double corr_dy = T_gicp(1, 3);
            double corr_dz = T_gicp(2, 3);
            double corr_trans = std::sqrt(corr_dx*corr_dx + corr_dy*corr_dy + corr_dz*corr_dz);

            // Check if correction is reasonable
            if (corr_trans > loop_closure_max_correction_m_) continue;

            // ── Loop closure detected! Apply correction ──────────────────────
            RCLCPP_INFO(get_logger(),
                "LOOP CLOSURE DETECTED: submap %d ↔ %d, fitness=%.4f, "
                "dist_traveled=%.2f m, correction=%.3f m",
                past_meta.submap_id, next_submap_id_ - 1,
                fitness, distance, corr_trans);

            {
                std::lock_guard<std::mutex> lock(tf_mutex_);
                T_map_odom_ = T_gicp * T_map_odom_;
            }

            loop_closure_count_++;

            // Publish loop closure marker
            if (loop_closure_pub_) {
                // Line connecting the two submaps
                visualization_msgs::msg::Marker loop_line;
                loop_line.header.frame_id = odom_frame_;
                loop_line.header.stamp = stamp;
                loop_line.ns = "loop_closures";
                loop_line.id = loop_closure_count_;
                loop_line.type = visualization_msgs::msg::Marker::LINE_STRIP;
                loop_line.action = visualization_msgs::msg::Marker::ADD;
                loop_line.pose.orientation.w = 1.0;
                loop_line.scale.x = 0.05;  // Line width
                loop_line.color.r = 0.0f;
                loop_line.color.g = 1.0f;
                loop_line.color.b = 0.0f;
                loop_line.color.a = 0.8f;

                geometry_msgs::msg::Point p1, p2;
                p1.x = past_meta.position(0);
                p1.y = past_meta.position(1);
                p1.z = past_meta.position(2);
                p2.x = current_pos(0);
                p2.y = current_pos(1);
                p2.z = current_pos(2);
                loop_line.points.push_back(p1);
                loop_line.points.push_back(p2);

                loop_markers.markers.push_back(loop_line);
            }
        }

        // Publish all loop closure markers
        if (loop_closure_pub_ && !loop_markers.markers.empty()) {
            loop_closure_pub_->publish(loop_markers);
        }
    }

    // ── Broadcast map→odom TF ───────────────────────────────────────────────
    void broadcastTF()
    {
        // Use the node's clock (sim-time aware with use_sim_time:=true)
        rclcpp::Time stamp = now();

        // Don't broadcast if clock hasn't started (time 0) or went backwards
        if (stamp.nanoseconds() == 0) return;
        if (stamp < last_tf_stamp_) return;
        last_tf_stamp_ = stamp;

        Eigen::Matrix4d T;
        {
            std::lock_guard<std::mutex> lock(tf_mutex_);
            T = T_map_odom_;
        }

        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = stamp;
        tf_msg.header.frame_id = map_frame_;
        tf_msg.child_frame_id = odom_frame_;

        tf_msg.transform.translation.x = T(0, 3);
        tf_msg.transform.translation.y = T(1, 3);
        tf_msg.transform.translation.z = T(2, 3);

        Eigen::Matrix3d R = T.block<3,3>(0,0);
        Eigen::Quaterniond q(R);
        q.normalize();
        tf_msg.transform.rotation.x = q.x();
        tf_msg.transform.rotation.y = q.y();
        tf_msg.transform.rotation.z = q.z();
        tf_msg.transform.rotation.w = q.w();

        tf_broadcaster_->sendTransform(tf_msg);
    }

    // ── Publish debug submap ────────────────────────────────────────────────
    void publishSubmap(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                       const rclcpp::Time& stamp)
    {
        if (!publish_submap_ || !submap_pub_) return;
        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(*cloud, msg);
        msg.header.frame_id = odom_frame_;
        msg.header.stamp = stamp;
        submap_pub_->publish(msg);
    }

    // ── Utility: Odometry msg → 4x4 matrix ─────────────────────────────────
    static Eigen::Matrix4d odomMsgToMatrix(
        const nav_msgs::msg::Odometry::ConstSharedPtr& msg)
    {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T(0, 3) = msg->pose.pose.position.x;
        T(1, 3) = msg->pose.pose.position.y;
        T(2, 3) = msg->pose.pose.position.z;

        Eigen::Quaterniond q(
            msg->pose.pose.orientation.w,
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z);
        T.block<3,3>(0,0) = q.normalized().toRotationMatrix();
        return T;
    }

    // ── Utility: TF → 4x4 float matrix ─────────────────────────────────────
    static Eigen::Matrix4f tfToMatrix4f(
        const geometry_msgs::msg::TransformStamped& tf)
    {
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T(0, 3) = static_cast<float>(tf.transform.translation.x);
        T(1, 3) = static_cast<float>(tf.transform.translation.y);
        T(2, 3) = static_cast<float>(tf.transform.translation.z);

        Eigen::Quaternionf q(
            static_cast<float>(tf.transform.rotation.w),
            static_cast<float>(tf.transform.rotation.x),
            static_cast<float>(tf.transform.rotation.y),
            static_cast<float>(tf.transform.rotation.z));
        T.block<3,3>(0,0) = q.normalized().toRotationMatrix();
        return T;
    }

    // ── Parameters ──────────────────────────────────────────────────────────
    std::string map_frame_, odom_frame_;
    int    accumulation_scans_;
    double keyframe_min_distance_;
    double voxel_leaf_size_;
    int    min_submap_points_;
    double max_correspondence_distance_;
    int    max_iterations_;
    double transformation_epsilon_;
    double euclidean_fitness_epsilon_;
    double max_fitness_score_;
    int    gicp_correspondence_randomness_;
    double tf_publish_hz_;
    bool   publish_submap_;
    double max_correction_m_;
    double max_correction_rad_;

    // ── State ───────────────────────────────────────────────────────────────
    Eigen::Matrix4d T_map_odom_;
    Eigen::Matrix4d keyframe_odom_pose_;
    Eigen::Matrix4d current_odom_pose_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr reference_submap_;
    int  scan_count_;
    bool has_reference_;
    bool has_odom_;
    rclcpp::Time last_odom_stamp_;
    rclcpp::Time last_tf_stamp_;

    // ── GICP ────────────────────────────────────────────────────────────────
    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp_;
    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp_stm_;

    // ── Scan-to-map ─────────────────────────────────────────────────────────
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_map_;
    std::deque<pcl::PointCloud<pcl::PointXYZ>> submap_history_;
    bool enable_scan_to_map_;
    int  global_map_max_submaps_;
    int  min_map_points_;

    // ── Loop closure ─────────────────────────────────────────────────────────
    bool enable_loop_closure_;
    double loop_closure_min_distance_;
    int    loop_closure_max_submaps_;
    double loop_closure_fitness_threshold_;
    double loop_closure_max_correction_m_;
    std::deque<SubmapData> submap_metadata_;
    int next_submap_id_;
    int loop_closure_count_;

    // ── TF ──────────────────────────────────────────────────────────────────
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // ── ROS interfaces ──────────────────────────────────────────────────────
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr inlier_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr submap_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr loop_closure_pub_;
    rclcpp::TimerBase::SharedPtr tf_timer_;

    // ── Thread safety ───────────────────────────────────────────────────────
    std::mutex odom_mutex_;
    std::mutex cloud_mutex_;
    std::mutex tf_mutex_;
};

// ── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RadarScanSlamNode>());
    rclcpp::shutdown();
    return 0;
}
