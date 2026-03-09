#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/exceptions.h>
#include <tf2/utils.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <mutex>
#include <deque>
#include <atomic>

namespace radar_mapping {

class IndoorRadarOccupancyGrid : public rclcpp::Node {
public:
  IndoorRadarOccupancyGrid();

private:
  // ========== STRUCTS ==========
  struct CellState {
    float log_odds = 0.0f;
    float confidence = 0.0f;
    rclcpp::Time last_seen_time;
    rclcpp::Time last_occupied_time;
    int observation_count = 0;
    int occupied_count = 0;
    int last_updated_frame = -1;
    float avg_rcs = 0.0f;
    bool is_dynamic = false;
    float velocity_accumulator = 0.0f;
    int dynamic_observations = 0;
    bool is_static_confirmed = false;
    bool is_behind_detection = false;
  };

  struct RadarDetection {
    double x_radar, y_radar, z_radar;
    double x_map, y_map;
    float range;
    float azimuth;
    float elevation;
    float rcs_u8;
    float velocity;
    int grid_idx;
    bool is_occluded = false;
  };

  // V5: Cached odometry pose for inter-scan compensation
  struct OdomPose {
    rclcpp::Time stamp;
    double x, y, z;
    double qx, qy, qz, qw;
    double vx, vy, vz;  // linear velocity in odom frame
  };

  struct Params {
    // Frames
    std::string map_frame{"odom"};
    std::string base_frame{"base_link"};
    std::string radar_frame{"ARS_548"};

    // Topics
    std::string input_topic{"/PointCloudDetectionMapping"};  // V5: mapping-specific cloud
    std::string grid_topic{"/map"};
    std::string filtered_cloud_topic{"/filtered_radar_cloud"};
    std::string odom_topic{"/fused_odom/odometry"};  // V5: fused odometry

    // Grid
    double grid_resolution{0.15};
    double grid_size_m{30.0};

    // V5: Adaptive grid growth — expands when robot nears any edge, preserving all data
    bool enable_moving_origin{true};
    double recenter_margin_m{5.0};  // expand when robot is this close to edge (m)
    double expand_step_m{10.0};     // how much to add per expansion (m)

    // V5: Ego-motion compensation
    bool enable_motion_compensation{true};
    int odom_buffer_size{200};  // ~4 seconds at 50 Hz

    // V5: Indoor multipath rejection
    bool enable_multipath_rejection{true};
    double multipath_range_ratio{1.8};     // reject points > ratio * nearest hit in same bin
    int multipath_min_near_hits{2};        // need this many near hits before rejecting far ones

    // Decay (time-based, uniform for all cell types)
    double decay_delay_sec{8.0};
    double decay_factor{0.999};      // Applied uniformly to all cells
    double decay_update_rate_hz{2.0};
    // NOTE: To enable tiered decay (static/dynamic/uncertain) for studying dynamic obstacles,
    // restore static_decay_factor and dynamic_decay_factor here, then uncomment conditional
    // logic in publishers.cpp::decay_unobserved_cells_() around line 30-37.

    // Static confirmation
    int min_observations_for_static{10};
    int dynamic_revoke_threshold{15};
    double occupied_protection_threshold{0.15};
    double occupied_protection_sec{5.0};

    // Confidence
    double confidence_publish_threshold{0.40};
    double confidence_increment{0.10};

    // Gaussian Inverse Sensor Model
    bool enable_gaussian_sensor_model{true};
    double gaussian_sigma_m{0.18};
    int gaussian_kernel_half{2};
    int gaussian_min_hits{3};       // min direct hits before spreading starts
    int gaussian_radial_half{1};    // radial tolerance: ±1 cell = ±0.07m
    int gaussian_tangential_half{3}; // tangential spread: ±3 cells = ±0.21m

    // Bayesian
    double p_occ{0.70};
    double p_free{0.45};
    double p0{0.5};
    double lo_min{-2.0};
    double lo_max{3.0};

    // Free space
    bool mark_free_space{true};
    double free_space_weight{0.08};
    double safety_margin_m{1.0};

    // Range
    double min_range{0.5};
    double max_range{15.0};

    // RCS
    std::string rcs_format{"uint8"};
    double rcs_min_db{-20.0};
    double rcs_max_db{20.0};
    double intensity_scale{0.35};
    double range_decay_factor{0.015};
    double min_rcs_threshold_db{-15.0};

    // Uint8 RCS thresholds for occlusion (angle-dependent)
    int min_rcs_u8_center{30};
    int min_rcs_u8_edge{60};
    int min_rcs_u8_high_edge{90};
    int min_rcs_u8_elevation{70};

    // Occlusion parameters
    bool enable_per_ray_occlusion{true};
    bool enable_map_occlusion{false};
    int map_occlusion_start_frame{500};
    double azimuth_bin_size_deg{2.0};
    double behind_range_margin_m{0.5};
    double behind_detection_weight{0.3};
    int min_hits_to_occlude{3};

    // Angle limits
    double max_azimuth_deg{70.0};
    double max_elevation_deg{20.0};
    double z_slice_min{-0.5};
    double z_slice_max{0.5};

    // Dynamic filter
    bool enable_dynamic_filter{true};
    double dynamic_velocity_threshold{0.8};
    bool enable_ego_compensation{true};

    // TF
    double tf_timeout_sec{0.1};
    double tf_stale_threshold_sec{0.5};

    // Map cleanliness
    int min_occupied_neighbors{1};      // suppress occupied cells with fewer than N occupied neighbors at publish time (0=off)
    bool enable_map_reset_on_loop{false}; // clear grid when bag loop is detected
    double map_reset_time_jump_sec{0.5};  // backward timestamp threshold to detect bag restart
  };

  // ========== MEMBER VARIABLES ==========
  Params params_;
  int current_frame_ = 0;
  nav_msgs::msg::OccupancyGrid grid_msg_;
  std::vector<CellState> cells_;
  int grid_width_, grid_height_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  double robot_x_in_map_ = 0.0;
  double robot_y_in_map_ = 0.0;
  double robot_vx_ = 0.0;
  double robot_vy_ = 0.0;
  double robot_yaw_ = 0.0;
  bool have_robot_pose_ = false;
  bool have_robot_velocity_ = false;
  rclcpp::Time last_tf_time_;

  // V5: Fused odometry state
  std::mutex odom_mutex_;
  std::deque<OdomPose> odom_buffer_;
  OdomPose latest_odom_;
  bool have_fused_odom_ = false;
  rclcpp::Time last_scan_time_;
  bool have_last_scan_time_ = false;

  // V5: Cached static transform (radar_frame -> base_link)
  geometry_msgs::msg::TransformStamped radar_to_base_cached_;
  bool have_radar_to_base_ = false;

  // V5: Diagnostic counters
  int frames_skipped_no_pose_ = 0;
  bool logged_first_registration_ = false;

  // V5: Grid origin tracking for moving origin
  double grid_origin_x_ = 0.0;
  double grid_origin_y_ = 0.0;

  // Trajectory tracking
  std::vector<geometry_msgs::msg::Point> trajectory_points_;
  double traj_last_x_ = 0.0, traj_last_y_ = 0.0;
  static constexpr double traj_min_dist_ = 0.05;  // record a point every 5cm

  std::atomic<bool> map_reset_requested_{false};
  std::mutex grid_mutex_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_grid_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_filtered_cloud_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_scale_bar_;
  rclcpp::TimerBase::SharedPtr decay_timer_;

  // ========== METHOD DECLARATIONS ==========

  // Parameters
  void declare_and_get_params_();

  // Grid utilities
  void init_grid_();
  void maybe_recenter_grid_();
  bool world_to_grid_(double x, double y, int& gx, int& gy) const;
  int grid_index_(int gx, int gy) const;
  float logit_(float p) const;
  float probability_from_logodds_(float lo) const;

  // Odometry handling
  void odom_callback_(const nav_msgs::msg::Odometry::SharedPtr msg);
  bool interpolate_odom_(const rclcpp::Time& stamp, OdomPose& result) const;
  void odom_pose_to_transform_(const OdomPose& pose, geometry_msgs::msg::TransformStamped& transform) const;
  bool transform_point_via_odom_(double x_radar, double y_radar, double z_radar,
                                 const OdomPose& odom_pose,
                                 const geometry_msgs::msg::TransformStamped& radar_to_base,
                                 double& x_map, double& y_map, double& z_map) const;

  // Sensor model & cell updates
  float normalize_rcs_(float raw_value) const;
  float get_min_rcs_u8_for_angle_(float azimuth_rad, float elevation_rad) const;
  float intensity_to_occupancy_(float raw_intensity, float range_m) const;
  void bresenham_(int x0, int y0, int x1, int y1, std::vector<std::pair<int,int>>& cells) const;
  float compensate_radial_velocity_(float v_measured, double x_radar, double y_radar) const;
  void apply_gaussian_occupied_update_(double hit_x, double hit_y,
                                       float base_lo_inc,
                                       const rclcpp::Time& stamp,
                                       bool is_occluded,
                                       float rcs_u8,
                                       float velocity_compensated);
  void mark_free_space_ray_(double robot_x, double robot_y, double hit_x, double hit_y,
                            const rclcpp::Time& stamp);

  // Occlusion & transforms
  bool get_transform_(const std::string& target_frame, const std::string& source_frame,
                      const rclcpp::Time& time, geometry_msgs::msg::TransformStamped& transform) const;
  bool transform_point_(double x_in, double y_in, double z_in,
                        const geometry_msgs::msg::TransformStamped& transform,
                        double& x_out, double& y_out, double& z_out);
  bool is_occluded_by_map_(double x_start, double y_start, double x_end, double y_end,
                           double range_end) const;
  void reject_multipath_(std::vector<RadarDetection>& detections) const;

  // Main processing
  void radar_callback_(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  // Publishing & decay
  void decay_unobserved_cells_();
  void publish_filtered_cloud_(const std::vector<RadarDetection>& detections,
                               const rclcpp::Time& stamp);
  void publish_grid_();
  void publish_scale_bar_();
};

}  // namespace radar_mapping
