/**
 * @file temporal_radar_occupancy_grid.cpp
 * @brief Radar Occupancy Grid with Fused Odometry Integration for Indoor Mapping
 *
 * NEW IN V5:
 * 1. Fused odometry subscription (/fused_odom/odometry) for accurate robot pose
 * 2. Odometry-based point cloud registration (uses fused pose instead of TF-only)
 * 3. Inter-scan ego-motion compensation (de-smears points using velocity between scans)
 * 4. Moving grid origin that follows the robot (re-centers when robot nears edge)
 * 5. Subscribes to /PointCloudDetectionMapping (separate filtering for mapping)
 * 6. Indoor-tuned sensor model (multipath rejection, wall-optimized probabilities)
 *
 * PREVIOUS V4:
 * - Per-ray occlusion, angle-dependent RCS, map-based occlusion, z-slice
 *
 * PREVIOUS V3:
 * - Safe locking, time-based decay, ego compensation, functional intensity_scale
 */

#include "temporal_radar_mapping/temporal_radar_occupancy_grid.hpp"

namespace radar_mapping {

IndoorRadarOccupancyGrid::IndoorRadarOccupancyGrid() : Node("temporal_radar_occupancy_grid") {
  declare_and_get_params_();
  init_grid_();

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Subscribe to mapping-specific filtered point cloud
  sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      params_.input_topic, rclcpp::SensorDataQoS(),
      std::bind(&IndoorRadarOccupancyGrid::radar_callback_, this, std::placeholders::_1));

  // V5: Subscribe to fused odometry for accurate pose & velocity
  sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      params_.odom_topic, rclcpp::SensorDataQoS(),
      std::bind(&IndoorRadarOccupancyGrid::odom_callback_, this, std::placeholders::_1));

  pub_grid_ = create_publisher<nav_msgs::msg::OccupancyGrid>(params_.grid_topic, 10);
  pub_filtered_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>(params_.filtered_cloud_topic, 10);
  pub_scale_bar_ = create_publisher<visualization_msgs::msg::MarkerArray>("/map_scale_bar", 10);

  decay_timer_ = create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(1000.0 / params_.decay_update_rate_hz)),
      std::bind(&IndoorRadarOccupancyGrid::decay_unobserved_cells_, this));

  RCLCPP_INFO(get_logger(), "=== Radar Occupancy Grid v5 (Fused Odom + Indoor) ===");
  RCLCPP_INFO(get_logger(), "  Input topic: %s", params_.input_topic.c_str());
  RCLCPP_INFO(get_logger(), "  Odom topic: %s", params_.odom_topic.c_str());
  RCLCPP_INFO(get_logger(), "  Filtered cloud topic: %s", params_.filtered_cloud_topic.c_str());
  RCLCPP_INFO(get_logger(), "  Ego-motion compensation: %s",
              params_.enable_motion_compensation ? "ON" : "OFF");
  RCLCPP_INFO(get_logger(), "  Moving grid origin: %s (recenter margin: %.1fm)",
              params_.enable_moving_origin ? "ON" : "OFF", params_.recenter_margin_m);
  RCLCPP_INFO(get_logger(), "  Indoor multipath rejection: %s",
              params_.enable_multipath_rejection ? "ON" : "OFF");
  RCLCPP_INFO(get_logger(), "  Azimuth binning: %.1f deg | Behind margin: %.2fm",
              params_.azimuth_bin_size_deg, params_.behind_range_margin_m);
  RCLCPP_INFO(get_logger(), "  RCS thresh (u8): center=%d edge=%d high-edge=%d",
              params_.min_rcs_u8_center, params_.min_rcs_u8_edge, params_.min_rcs_u8_high_edge);
  RCLCPP_INFO(get_logger(), "  Z-slice range: [%.2f, %.2f] m",
              params_.z_slice_min, params_.z_slice_max);
  RCLCPP_INFO(get_logger(), "  Map occlusion: %s (enable after %d frames)",
              params_.enable_map_occlusion ? "YES" : "NO", params_.map_occlusion_start_frame);
  RCLCPP_INFO(get_logger(), "  Gaussian sensor model: %s (sigma=%.2fm, tang=+/-%d rad=+/-%d minHits=%d)",
              params_.enable_gaussian_sensor_model ? "ENABLED" : "OFF",
              params_.gaussian_sigma_m,
              params_.gaussian_tangential_half,
              params_.gaussian_radial_half,
              params_.gaussian_min_hits);
}


void IndoorRadarOccupancyGrid::declare_and_get_params_() {
  // Frames
  declare_parameter("map_frame", params_.map_frame);
  declare_parameter("base_frame", params_.base_frame);
  declare_parameter("radar_frame", params_.radar_frame);

  // Topics
  declare_parameter("input_topic", params_.input_topic);
  declare_parameter("grid_topic", params_.grid_topic);
  declare_parameter("filtered_cloud_topic", params_.filtered_cloud_topic);
  declare_parameter("odom_topic", params_.odom_topic);

  // Grid
  declare_parameter("grid_resolution", params_.grid_resolution);
  declare_parameter("grid_size_m", params_.grid_size_m);

  // V5: Adaptive grid growth
  declare_parameter("enable_moving_origin", params_.enable_moving_origin);
  declare_parameter("recenter_margin_m", params_.recenter_margin_m);
  declare_parameter("expand_step_m", params_.expand_step_m);

  // V5: Motion compensation
  declare_parameter("enable_motion_compensation", params_.enable_motion_compensation);
  declare_parameter("odom_buffer_size", params_.odom_buffer_size);

  // V5: Indoor multipath
  declare_parameter("enable_multipath_rejection", params_.enable_multipath_rejection);
  declare_parameter("multipath_range_ratio", params_.multipath_range_ratio);
  declare_parameter("multipath_min_near_hits", params_.multipath_min_near_hits);

  // Decay (uniform for all cell types)
  declare_parameter("decay_delay_sec", params_.decay_delay_sec);
  declare_parameter("decay_factor", params_.decay_factor);
  declare_parameter("decay_update_rate_hz", params_.decay_update_rate_hz);

  // Static confirmation
  declare_parameter("min_observations_for_static", params_.min_observations_for_static);
  declare_parameter("dynamic_revoke_threshold", params_.dynamic_revoke_threshold);
  declare_parameter("occupied_protection_threshold", params_.occupied_protection_threshold);
  declare_parameter("occupied_protection_sec", params_.occupied_protection_sec);

  // Confidence
  declare_parameter("confidence_publish_threshold", params_.confidence_publish_threshold);
  declare_parameter("confidence_increment", params_.confidence_increment);

  // Gaussian Inverse Sensor Model
  declare_parameter("enable_gaussian_sensor_model", params_.enable_gaussian_sensor_model);
  declare_parameter("gaussian_sigma_m", params_.gaussian_sigma_m);
  declare_parameter("gaussian_kernel_half", params_.gaussian_kernel_half);
  declare_parameter("gaussian_min_hits", params_.gaussian_min_hits);
  declare_parameter("gaussian_radial_half", params_.gaussian_radial_half);
  declare_parameter("gaussian_tangential_half", params_.gaussian_tangential_half);

  // Bayesian
  declare_parameter("p_occ", params_.p_occ);
  declare_parameter("p_free", params_.p_free);
  declare_parameter("p0", params_.p0);
  declare_parameter("lo_min", params_.lo_min);
  declare_parameter("lo_max", params_.lo_max);

  // Free space
  declare_parameter("mark_free_space", params_.mark_free_space);
  declare_parameter("free_space_weight", params_.free_space_weight);
  declare_parameter("safety_margin_m", params_.safety_margin_m);

  // Range
  declare_parameter("min_range", params_.min_range);
  declare_parameter("max_range", params_.max_range);

  // RCS
  declare_parameter("rcs_format", params_.rcs_format);
  declare_parameter("rcs_min_db", params_.rcs_min_db);
  declare_parameter("rcs_max_db", params_.rcs_max_db);
  declare_parameter("intensity_scale", params_.intensity_scale);
  declare_parameter("range_decay_factor", params_.range_decay_factor);
  declare_parameter("min_rcs_threshold_db", params_.min_rcs_threshold_db);

  // Occlusion params
  declare_parameter("min_rcs_u8_center", params_.min_rcs_u8_center);
  declare_parameter("min_rcs_u8_edge", params_.min_rcs_u8_edge);
  declare_parameter("min_rcs_u8_high_edge", params_.min_rcs_u8_high_edge);
  declare_parameter("min_rcs_u8_elevation", params_.min_rcs_u8_elevation);
  declare_parameter("enable_per_ray_occlusion", params_.enable_per_ray_occlusion);
  declare_parameter("enable_map_occlusion", params_.enable_map_occlusion);
  declare_parameter("map_occlusion_start_frame", params_.map_occlusion_start_frame);
  declare_parameter("azimuth_bin_size_deg", params_.azimuth_bin_size_deg);
  declare_parameter("behind_range_margin_m", params_.behind_range_margin_m);
  declare_parameter("behind_detection_weight", params_.behind_detection_weight);
  declare_parameter("min_hits_to_occlude", params_.min_hits_to_occlude);

  // Angle limits and z-slice
  declare_parameter("max_azimuth_deg", params_.max_azimuth_deg);
  declare_parameter("max_elevation_deg", params_.max_elevation_deg);
  declare_parameter("z_slice_min", params_.z_slice_min);
  declare_parameter("z_slice_max", params_.z_slice_max);

  // Dynamic filter
  declare_parameter("enable_dynamic_filter", params_.enable_dynamic_filter);
  declare_parameter("dynamic_velocity_threshold", params_.dynamic_velocity_threshold);
  declare_parameter("enable_ego_compensation", params_.enable_ego_compensation);

  // TF
  declare_parameter("tf_timeout_sec", params_.tf_timeout_sec);
  declare_parameter("tf_stale_threshold_sec", params_.tf_stale_threshold_sec);

  // Map cleanliness
  declare_parameter("min_occupied_neighbors", params_.min_occupied_neighbors);
  declare_parameter("enable_map_reset_on_loop", params_.enable_map_reset_on_loop);
  declare_parameter("map_reset_time_jump_sec", params_.map_reset_time_jump_sec);

  // GET all parameters
  params_.map_frame = get_parameter("map_frame").as_string();
  params_.base_frame = get_parameter("base_frame").as_string();
  params_.radar_frame = get_parameter("radar_frame").as_string();
  params_.input_topic = get_parameter("input_topic").as_string();
  params_.grid_topic = get_parameter("grid_topic").as_string();
  params_.filtered_cloud_topic = get_parameter("filtered_cloud_topic").as_string();
  params_.odom_topic = get_parameter("odom_topic").as_string();
  params_.grid_resolution = get_parameter("grid_resolution").as_double();
  params_.grid_size_m = get_parameter("grid_size_m").as_double();
  params_.enable_moving_origin = get_parameter("enable_moving_origin").as_bool();
  params_.recenter_margin_m = get_parameter("recenter_margin_m").as_double();
  params_.expand_step_m = get_parameter("expand_step_m").as_double();
  params_.enable_motion_compensation = get_parameter("enable_motion_compensation").as_bool();
  params_.odom_buffer_size = get_parameter("odom_buffer_size").as_int();
  params_.enable_multipath_rejection = get_parameter("enable_multipath_rejection").as_bool();
  params_.multipath_range_ratio = get_parameter("multipath_range_ratio").as_double();
  params_.multipath_min_near_hits = get_parameter("multipath_min_near_hits").as_int();
  params_.decay_delay_sec = get_parameter("decay_delay_sec").as_double();
  params_.decay_factor = get_parameter("decay_factor").as_double();
  params_.decay_update_rate_hz = get_parameter("decay_update_rate_hz").as_double();
  params_.min_observations_for_static = get_parameter("min_observations_for_static").as_int();
  params_.dynamic_revoke_threshold = get_parameter("dynamic_revoke_threshold").as_int();
  params_.occupied_protection_threshold = get_parameter("occupied_protection_threshold").as_double();
  params_.occupied_protection_sec = get_parameter("occupied_protection_sec").as_double();
  params_.confidence_publish_threshold = get_parameter("confidence_publish_threshold").as_double();
  params_.confidence_increment = get_parameter("confidence_increment").as_double();
  params_.enable_gaussian_sensor_model = get_parameter("enable_gaussian_sensor_model").as_bool();
  params_.gaussian_sigma_m = get_parameter("gaussian_sigma_m").as_double();
  params_.gaussian_kernel_half = get_parameter("gaussian_kernel_half").as_int();
  params_.gaussian_min_hits = get_parameter("gaussian_min_hits").as_int();
  params_.gaussian_radial_half = get_parameter("gaussian_radial_half").as_int();
  params_.gaussian_tangential_half = get_parameter("gaussian_tangential_half").as_int();
  params_.p_occ = get_parameter("p_occ").as_double();
  params_.p_free = get_parameter("p_free").as_double();
  params_.p0 = get_parameter("p0").as_double();
  params_.lo_min = get_parameter("lo_min").as_double();
  params_.lo_max = get_parameter("lo_max").as_double();
  params_.mark_free_space = get_parameter("mark_free_space").as_bool();
  params_.free_space_weight = get_parameter("free_space_weight").as_double();
  params_.safety_margin_m = get_parameter("safety_margin_m").as_double();
  params_.min_range = get_parameter("min_range").as_double();
  params_.max_range = get_parameter("max_range").as_double();
  params_.rcs_format = get_parameter("rcs_format").as_string();
  params_.rcs_min_db = get_parameter("rcs_min_db").as_double();
  params_.rcs_max_db = get_parameter("rcs_max_db").as_double();
  params_.intensity_scale = get_parameter("intensity_scale").as_double();
  params_.range_decay_factor = get_parameter("range_decay_factor").as_double();
  params_.min_rcs_threshold_db = get_parameter("min_rcs_threshold_db").as_double();

  params_.min_rcs_u8_center = get_parameter("min_rcs_u8_center").as_int();
  params_.min_rcs_u8_edge = get_parameter("min_rcs_u8_edge").as_int();
  params_.min_rcs_u8_high_edge = get_parameter("min_rcs_u8_high_edge").as_int();
  params_.min_rcs_u8_elevation = get_parameter("min_rcs_u8_elevation").as_int();
  params_.enable_per_ray_occlusion = get_parameter("enable_per_ray_occlusion").as_bool();
  params_.enable_map_occlusion = get_parameter("enable_map_occlusion").as_bool();
  params_.map_occlusion_start_frame = get_parameter("map_occlusion_start_frame").as_int();
  params_.azimuth_bin_size_deg = get_parameter("azimuth_bin_size_deg").as_double();
  params_.behind_range_margin_m = get_parameter("behind_range_margin_m").as_double();
  params_.behind_detection_weight = get_parameter("behind_detection_weight").as_double();
  params_.min_hits_to_occlude = get_parameter("min_hits_to_occlude").as_int();
  params_.max_azimuth_deg = get_parameter("max_azimuth_deg").as_double();
  params_.max_elevation_deg = get_parameter("max_elevation_deg").as_double();
  params_.z_slice_min = get_parameter("z_slice_min").as_double();
  params_.z_slice_max = get_parameter("z_slice_max").as_double();

  params_.enable_dynamic_filter = get_parameter("enable_dynamic_filter").as_bool();
  params_.dynamic_velocity_threshold = get_parameter("dynamic_velocity_threshold").as_double();
  params_.enable_ego_compensation = get_parameter("enable_ego_compensation").as_bool();
  params_.tf_timeout_sec = get_parameter("tf_timeout_sec").as_double();
  params_.tf_stale_threshold_sec = get_parameter("tf_stale_threshold_sec").as_double();
  params_.min_occupied_neighbors = get_parameter("min_occupied_neighbors").as_int();
  params_.enable_map_reset_on_loop = get_parameter("enable_map_reset_on_loop").as_bool();
  params_.map_reset_time_jump_sec = get_parameter("map_reset_time_jump_sec").as_double();
}

}  // namespace radar_mapping


int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<radar_mapping::IndoorRadarOccupancyGrid>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
  executor.add_node(node);
  RCLCPP_INFO(node->get_logger(), "Starting Radar Occupancy Grid v5 (Fused Odom + Indoor)");
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
