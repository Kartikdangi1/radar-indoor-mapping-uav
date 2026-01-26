/**
 * @file temporal_radar_occupancy_grid_v2.cpp
 * @brief FIXED Radar Occupancy Grid with TF Transforms and Map Accumulation
 * 
 * KEY FIXES:
 * 1. TF2 transforms: Points are transformed from radar frame to map frame
 * 2. Global map accumulation: Grid is fixed in map frame, robot moves through it
 * 3. Improved persistence: Static objects stay longer, smarter decay
 * 4. Protected free-space: Won't erase recently-occupied cells
 * 5. Proper RCS/SNR handling for ARS_548 UINT8 format
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/exceptions.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <deque>
#include <mutex>

class IndoorRadarOccupancyGrid : public rclcpp::Node {
public:
  IndoorRadarOccupancyGrid() : Node("temporal_radar_occupancy_grid"), current_frame_(0) {
    declare_and_get_params_();
    init_grid_();
    
    // Initialize TF2
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      params_.input_topic, rclcpp::SensorDataQoS(),
      std::bind(&IndoorRadarOccupancyGrid::radar_callback_, this, std::placeholders::_1));
    
    pub_grid_ = create_publisher<nav_msgs::msg::OccupancyGrid>(params_.grid_topic, 10);
    pub_debug_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>("/radar/transformed_points", 10);
    
    decay_timer_ = create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(1000.0 / params_.decay_update_rate_hz)),
      std::bind(&IndoorRadarOccupancyGrid::decay_unobserved_cells_, this));
    
    RCLCPP_INFO(get_logger(), "=== Radar Occupancy Grid v2 (TF + Persistence) ===");
    RCLCPP_INFO(get_logger(), "  Input: %s", params_.input_topic.c_str());
    RCLCPP_INFO(get_logger(), "  Output: %s in frame '%s'", params_.grid_topic.c_str(), params_.map_frame.c_str());
    RCLCPP_INFO(get_logger(), "  Radar frame: %s -> Map frame: %s", params_.radar_frame.c_str(), params_.map_frame.c_str());
    RCLCPP_INFO(get_logger(), "  Grid: %.1fm x %.1fm @ %.2fm resolution (%d x %d cells)", 
                params_.grid_size_m, params_.grid_size_m, params_.grid_resolution, grid_width_, grid_height_);
    RCLCPP_INFO(get_logger(), "  Free space: %s | Decay: frames_before=%d, factor=%.3f",
                params_.mark_free_space ? "ON" : "OFF", params_.frames_before_decay, params_.decay_factor);
    RCLCPP_INFO(get_logger(), "  Static protection: min_obs=%d, protect_threshold=%.2f",
                params_.min_observations_for_static, params_.occupied_protection_threshold);
  }

private:
  // ========== CELL STATE ==========
  struct CellState {
    float log_odds = 0.0f;
    float confidence = 0.0f;
    int last_seen_frame = -1;
    int last_occupied_frame = -1;    // NEW: Track when last seen as occupied
    int observation_count = 0;
    int occupied_count = 0;          // NEW: Count of occupied observations
    float observation_consistency = 1.0f;
    float weighted_confidence = 0.0f;
    int detection_count_this_frame = 0;
    bool is_behind_detection = false;
    float avg_rcs = 0.0f;            // Renamed from avg_snr for clarity
    bool is_dynamic = false;
    float velocity_accumulator = 0.0f;
    int dynamic_observations = 0;
    bool is_static_confirmed = false; // NEW: Cell confirmed as static obstacle
  };

  struct Params {
    // Frame configuration - NEW
    std::string map_frame{"odom"};           // Fixed frame for the map
    std::string base_frame{"base_link"};     // Robot base frame
    std::string radar_frame{"ARS_548"};      // Radar sensor frame
    
    // Topics
    std::string input_topic{"/PointCloudDetection"};
    std::string grid_topic{"/map"};
    
    // Grid parameters
    double grid_resolution{0.25};
    double grid_size_m{50.0};
    
    // Temporal parameters - TUNED FOR PERSISTENCE
    double temporal_decay_rate{0.01};        // Slower decay
    int frames_before_decay{100};            // Much longer before decay starts
    double decay_factor{0.995};              // Very slow decay
    double decay_update_rate_hz{1.0};        // Less frequent decay
    
    // Static object protection - NEW
    int min_observations_for_static{3};      // Min observations to be "static"
    double occupied_protection_threshold{0.5}; // Don't free-space cells above this log_odds
    int occupied_protection_frames{50};      // Don't clear recently-occupied cells
    double static_decay_factor{0.999};       // Almost no decay for confirmed static
    
    // Confidence parameters
    double confidence_publish_threshold{0.15}; // Lower threshold to see more
    double confidence_increment{0.2};
    
    // Bayesian parameters
    double p_occ{0.75};
    double p_free{0.4};
    double p0{0.5};
    double lo_min{-2.0};
    double lo_max{3.0};                      // Asymmetric: harder to clear than occupy
    
    // Free space parameters - CONSERVATIVE
    bool mark_free_space{true};
    double free_space_weight{0.3};           // Weaker free-space evidence
    double safety_margin_m{0.5};             // Larger margin
    
    // Range parameters
    double min_range{1.0};
    double max_range{40.0};
    
    // RCS handling for ARS_548
    std::string rcs_format{"uint8"};
    double rcs_min_db{-20.0};
    double rcs_max_db{40.0};
    double intensity_scale{0.08};
    
    // Dynamic filter
    bool enable_dynamic_filter{true};
    double dynamic_velocity_threshold{2.0};
    double dynamic_decay_factor{0.95};
    
    // TF timeout
    double tf_timeout_sec{0.1};
  };

  Params params_;
  int current_frame_;
  
  nav_msgs::msg::OccupancyGrid grid_msg_;
  std::vector<CellState> cells_;
  int grid_width_;
  int grid_height_;
  
  // TF2
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  
  // Robot position in map frame (for ray casting origin)
  double robot_x_in_map_ = 0.0;
  double robot_y_in_map_ = 0.0;
  bool have_robot_pose_ = false;
  
  std::mutex grid_mutex_;
  
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_grid_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_debug_cloud_;
  rclcpp::TimerBase::SharedPtr decay_timer_;

  // ========== PARAMETER DECLARATION ==========
  void declare_and_get_params_() {
    // Frame parameters - NEW
    declare_parameter("map_frame", params_.map_frame);
    declare_parameter("base_frame", params_.base_frame);
    declare_parameter("radar_frame", params_.radar_frame);
    
    // Topics
    declare_parameter("input_topic", params_.input_topic);
    declare_parameter("grid_topic", params_.grid_topic);
    
    // Grid
    declare_parameter("grid_resolution", params_.grid_resolution);
    declare_parameter("grid_size_m", params_.grid_size_m);
    
    // Temporal
    declare_parameter("temporal_decay_rate", params_.temporal_decay_rate);
    declare_parameter("frames_before_decay", params_.frames_before_decay);
    declare_parameter("decay_factor", params_.decay_factor);
    declare_parameter("decay_update_rate_hz", params_.decay_update_rate_hz);
    
    // Static protection - NEW
    declare_parameter("min_observations_for_static", params_.min_observations_for_static);
    declare_parameter("occupied_protection_threshold", params_.occupied_protection_threshold);
    declare_parameter("occupied_protection_frames", params_.occupied_protection_frames);
    declare_parameter("static_decay_factor", params_.static_decay_factor);
    
    // Confidence
    declare_parameter("confidence_publish_threshold", params_.confidence_publish_threshold);
    declare_parameter("confidence_increment", params_.confidence_increment);
    
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
    
    // Dynamic
    declare_parameter("enable_dynamic_filter", params_.enable_dynamic_filter);
    declare_parameter("dynamic_velocity_threshold", params_.dynamic_velocity_threshold);
    declare_parameter("dynamic_decay_factor", params_.dynamic_decay_factor);
    
    // TF
    declare_parameter("tf_timeout_sec", params_.tf_timeout_sec);
    
    // GET parameters
    params_.map_frame = get_parameter("map_frame").as_string();
    params_.base_frame = get_parameter("base_frame").as_string();
    params_.radar_frame = get_parameter("radar_frame").as_string();
    params_.input_topic = get_parameter("input_topic").as_string();
    params_.grid_topic = get_parameter("grid_topic").as_string();
    params_.grid_resolution = get_parameter("grid_resolution").as_double();
    params_.grid_size_m = get_parameter("grid_size_m").as_double();
    params_.temporal_decay_rate = get_parameter("temporal_decay_rate").as_double();
    params_.frames_before_decay = get_parameter("frames_before_decay").as_int();
    params_.decay_factor = get_parameter("decay_factor").as_double();
    params_.decay_update_rate_hz = get_parameter("decay_update_rate_hz").as_double();
    params_.min_observations_for_static = get_parameter("min_observations_for_static").as_int();
    params_.occupied_protection_threshold = get_parameter("occupied_protection_threshold").as_double();
    params_.occupied_protection_frames = get_parameter("occupied_protection_frames").as_int();
    params_.static_decay_factor = get_parameter("static_decay_factor").as_double();
    params_.confidence_publish_threshold = get_parameter("confidence_publish_threshold").as_double();
    params_.confidence_increment = get_parameter("confidence_increment").as_double();
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
    params_.enable_dynamic_filter = get_parameter("enable_dynamic_filter").as_bool();
    params_.dynamic_velocity_threshold = get_parameter("dynamic_velocity_threshold").as_double();
    params_.dynamic_decay_factor = get_parameter("dynamic_decay_factor").as_double();
    params_.tf_timeout_sec = get_parameter("tf_timeout_sec").as_double();
  }

  void init_grid_() {
    grid_width_ = static_cast<int>(std::ceil(params_.grid_size_m / params_.grid_resolution));
    grid_height_ = grid_width_;
    const size_t num_cells = static_cast<size_t>(grid_width_) * static_cast<size_t>(grid_height_);
    
    cells_.resize(num_cells);
    
    grid_msg_.info.resolution = static_cast<float>(params_.grid_resolution);
    grid_msg_.info.width = static_cast<uint32_t>(grid_width_);
    grid_msg_.info.height = static_cast<uint32_t>(grid_height_);
    
    // Grid origin is fixed in map frame - centered at (0,0)
    grid_msg_.info.origin.position.x = -params_.grid_size_m / 2.0;
    grid_msg_.info.origin.position.y = -params_.grid_size_m / 2.0;
    grid_msg_.info.origin.position.z = 0.0;
    grid_msg_.info.origin.orientation.w = 1.0;
    
    // CRITICAL: Grid is in map frame, not base_link
    grid_msg_.header.frame_id = params_.map_frame;
    
    grid_msg_.data.resize(num_cells, -1);
  }

  // ========== COORDINATE TRANSFORMS ==========
  inline bool world_to_grid_(double x, double y, int &gx, int &gy) const {
    const double ox = grid_msg_.info.origin.position.x;
    const double oy = grid_msg_.info.origin.position.y;
    const double r = params_.grid_resolution;
    gx = static_cast<int>(std::floor((x - ox) / r));
    gy = static_cast<int>(std::floor((y - oy) / r));
    return (gx >= 0 && gy >= 0 && gx < grid_width_ && gy < grid_height_);
  }

  inline int grid_index_(int gx, int gy) const {
    return gy * grid_width_ + gx;
  }

  inline float logit_(float p) const {
    p = std::clamp(p, 0.01f, 0.99f);
    return std::log(p / (1.0f - p));
  }

  inline float probability_from_logodds_(float lo) const {
    return 1.0f / (1.0f + std::exp(-lo));
  }

  // RCS normalization for UINT8 format
  float normalize_rcs_(float raw_value) const {
    if (params_.rcs_format == "uint8") {
      float range = static_cast<float>(params_.rcs_max_db - params_.rcs_min_db);
      return (raw_value / 255.0f) * range + static_cast<float>(params_.rcs_min_db);
    }
    return raw_value;
  }

  float intensity_to_occupancy_(float raw_intensity) const {
    float rcs_db = normalize_rcs_(raw_intensity);
    float normalized = (rcs_db - static_cast<float>(params_.rcs_min_db)) / 
                       static_cast<float>(params_.rcs_max_db - params_.rcs_min_db);
    normalized = std::clamp(normalized, 0.0f, 1.0f);
    float prob = 0.55f + normalized * 0.35f;  // Range: 0.55 to 0.90
    return prob;
  }

  // Bresenham line for ray casting
  void bresenham_(int x0, int y0, int x1, int y1, std::vector<std::pair<int,int>> &cells) const {
    cells.clear();
    int dx = std::abs(x1 - x0);
    int sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0);
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    int x = x0, y = y0;
    
    while (true) {
      if (x >= 0 && y >= 0 && x < grid_width_ && y < grid_height_) {
        cells.push_back({x, y});
      }
      if (x == x1 && y == y1) break;
      int e2 = 2 * err;
      if (e2 >= dy) { err += dy; x += sx; }
      if (e2 <= dx) { err += dx; y += sy; }
    }
  }

  // ========== TF LOOKUP ==========
  bool get_transform_(const std::string& target_frame, const std::string& source_frame,
                      const rclcpp::Time& time, geometry_msgs::msg::TransformStamped& transform) {
    try {
      transform = tf_buffer_->lookupTransform(
        target_frame, source_frame, time,
        rclcpp::Duration::from_seconds(params_.tf_timeout_sec));
      return true;
    } catch (tf2::TransformException& ex) {
      // Try with latest available
      try {
        transform = tf_buffer_->lookupTransform(
          target_frame, source_frame, tf2::TimePointZero);
        return true;
      } catch (tf2::TransformException& ex2) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "TF %s -> %s failed: %s", source_frame.c_str(), target_frame.c_str(), ex2.what());
        return false;
      }
    }
  }

  // Transform a point from source frame to target frame
  bool transform_point_(double x_in, double y_in, double z_in,
                        const geometry_msgs::msg::TransformStamped& transform,
                        double& x_out, double& y_out, double& z_out) {
    geometry_msgs::msg::PointStamped pt_in, pt_out;
    pt_in.point.x = x_in;
    pt_in.point.y = y_in;
    pt_in.point.z = z_in;
    
    try {
      tf2::doTransform(pt_in, pt_out, transform);
      x_out = pt_out.point.x;
      y_out = pt_out.point.y;
      z_out = pt_out.point.z;
      return true;
    } catch (tf2::TransformException& ex) {
      return false;
    }
  }

  // ========== CELL UPDATE (with static protection) ==========
  void update_cell_occupied_(int idx, float rcs, float velocity) {
    if (idx < 0 || idx >= static_cast<int>(cells_.size())) return;
    
    std::lock_guard<std::mutex> lock(grid_mutex_);
    CellState &cell = cells_[idx];
    
    // Dynamic classification
    if (params_.enable_dynamic_filter) {
      cell.velocity_accumulator = 0.9f * cell.velocity_accumulator + 0.1f * std::abs(velocity);
      if (cell.velocity_accumulator > params_.dynamic_velocity_threshold) {
        cell.is_dynamic = true;
        cell.dynamic_observations++;
      } else if (cell.dynamic_observations > 0) {
        cell.dynamic_observations--;
        if (cell.dynamic_observations == 0) cell.is_dynamic = false;
      }
    }
    
    // Update RCS average
    float rcs_db = normalize_rcs_(rcs);
    cell.avg_rcs = 0.8f * cell.avg_rcs + 0.2f * rcs_db;
    
    // Compute occupancy probability from RCS
    float occ_prob = intensity_to_occupancy_(rcs);
    float lo_inc = logit_(occ_prob) - logit_(static_cast<float>(params_.p0));
    
    // Update log-odds
    cell.log_odds += lo_inc;
    cell.log_odds = std::clamp(cell.log_odds, 
                                static_cast<float>(params_.lo_min), 
                                static_cast<float>(params_.lo_max));
    
    // Update confidence
    cell.confidence = std::min(1.0f, cell.confidence + static_cast<float>(params_.confidence_increment));
    
    // Update observation counts
    cell.observation_count++;
    cell.occupied_count++;
    cell.last_seen_frame = current_frame_;
    cell.last_occupied_frame = current_frame_;
    cell.detection_count_this_frame++;
    
    // Mark as static if enough consistent occupied observations
    if (cell.occupied_count >= params_.min_observations_for_static && !cell.is_dynamic) {
      cell.is_static_confirmed = true;
    }
  }

  // ========== FREE SPACE MARKING (with protection) ==========
  void mark_free_space_ray_(double robot_x, double robot_y, double hit_x, double hit_y) {
    if (!params_.mark_free_space) return;
    
    int sx, sy, ex, ey;
    if (!world_to_grid_(robot_x, robot_y, sx, sy)) return;
    if (!world_to_grid_(hit_x, hit_y, ex, ey)) return;
    
    double hit_range = std::sqrt((hit_x - robot_x) * (hit_x - robot_x) + 
                                  (hit_y - robot_y) * (hit_y - robot_y));
    
    std::vector<std::pair<int,int>> ray_cells;
    bresenham_(sx, sy, ex, ey, ray_cells);
    if (ray_cells.empty()) return;
    
    const float lo_free_inc = logit_(static_cast<float>(params_.p_free)) - 
                               logit_(static_cast<float>(params_.p0));
    const float weighted_free = lo_free_inc * static_cast<float>(params_.free_space_weight);
    
    std::lock_guard<std::mutex> lock(grid_mutex_);
    
    for (size_t i = 0; i < ray_cells.size() - 1; ++i) {  // Stop before hit cell
      int gx = ray_cells[i].first;
      int gy = ray_cells[i].second;
      
      double cell_x = grid_msg_.info.origin.position.x + (gx + 0.5) * params_.grid_resolution;
      double cell_y = grid_msg_.info.origin.position.y + (gy + 0.5) * params_.grid_resolution;
      double cell_range = std::sqrt((cell_x - robot_x) * (cell_x - robot_x) + 
                                     (cell_y - robot_y) * (cell_y - robot_y));
      
      // Only mark as free if well before the hit point
      if (cell_range > hit_range - params_.safety_margin_m) {
        break;
      }
      
      int idx = grid_index_(gx, gy);
      CellState &cell = cells_[idx];
      
      // PROTECTION: Don't clear cells that are likely occupied
      if (cell.log_odds > params_.occupied_protection_threshold) {
        continue;  // Skip - this cell is probably occupied
      }
      
      // PROTECTION: Don't clear recently-occupied cells
      if (cell.last_occupied_frame >= 0 && 
          (current_frame_ - cell.last_occupied_frame) < params_.occupied_protection_frames) {
        continue;  // Skip - recently saw something here
      }
      
      // PROTECTION: Don't clear confirmed static cells
      if (cell.is_static_confirmed) {
        continue;
      }
      
      // Safe to mark as free
      cell.log_odds += weighted_free;
      cell.log_odds = std::max(cell.log_odds, static_cast<float>(params_.lo_min));
      cell.last_seen_frame = current_frame_;
      cell.observation_count++;
    }
  }

  // ========== DECAY (with static protection) ==========
  void decay_unobserved_cells_() {
    std::lock_guard<std::mutex> lock(grid_mutex_);
    
    for (size_t i = 0; i < cells_.size(); ++i) {
      CellState &cell = cells_[i];
      
      if (cell.last_seen_frame < 0) continue;
      
      int age = current_frame_ - cell.last_seen_frame;
      
      if (age <= params_.frames_before_decay) continue;
      
      // Choose decay factor based on cell type
      float decay;
      if (cell.is_static_confirmed) {
        decay = static_cast<float>(params_.static_decay_factor);  // Almost no decay
      } else if (cell.is_dynamic) {
        decay = static_cast<float>(params_.dynamic_decay_factor); // Fast decay
      } else {
        decay = static_cast<float>(params_.decay_factor);         // Normal decay
      }
      
      // Only decay confidence, not log_odds directly for occupied cells
      if (cell.log_odds > 0) {
        // Occupied cell: decay confidence but preserve log_odds longer
        cell.confidence *= decay;
        if (cell.confidence < 0.1f) {
          cell.log_odds *= decay;  // Only now start decaying occupancy
        }
      } else {
        // Free cell: can decay normally
        cell.log_odds *= decay;
        cell.confidence *= decay;
      }
      
      // Reset if fully decayed
      if (std::abs(cell.log_odds) < 0.01f && cell.confidence < 0.05f) {
        cell.log_odds = 0.0f;
        cell.confidence = 0.0f;
        cell.last_seen_frame = -1;
        cell.last_occupied_frame = -1;
        cell.observation_count = 0;
        cell.occupied_count = 0;
        cell.is_dynamic = false;
        cell.is_static_confirmed = false;
      }
    }
  }

  // ========== MAIN CALLBACK ==========
  void radar_callback_(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    current_frame_++;
    
    // Get transform from radar frame to map frame
    std::string source_frame = msg->header.frame_id;
    if (source_frame.empty()) {
      source_frame = params_.radar_frame;
    }
    
    geometry_msgs::msg::TransformStamped radar_to_map;
    if (!get_transform_(params_.map_frame, source_frame, msg->header.stamp, radar_to_map)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, 
        "Cannot get TF %s -> %s, skipping frame", source_frame.c_str(), params_.map_frame.c_str());
      return;
    }
    
    // Get robot position in map frame (for ray casting origin)
    geometry_msgs::msg::TransformStamped base_to_map;
    if (get_transform_(params_.map_frame, params_.base_frame, msg->header.stamp, base_to_map)) {
      robot_x_in_map_ = base_to_map.transform.translation.x;
      robot_y_in_map_ = base_to_map.transform.translation.y;
      have_robot_pose_ = true;
    }
    
    // Find field offsets
    int offset_x = -1, offset_y = -1, offset_z = -1, offset_rcs = -1, offset_v = -1;
    uint8_t rcs_datatype = sensor_msgs::msg::PointField::FLOAT32;
    
    for (const auto &field : msg->fields) {
      if (field.name == "x") offset_x = field.offset;
      else if (field.name == "y") offset_y = field.offset;
      else if (field.name == "z") offset_z = field.offset;
      else if (field.name == "RCS" || field.name == "rcs" || field.name == "intensity") {
        offset_rcs = field.offset;
        rcs_datatype = field.datatype;
      }
      else if (field.name == "v" || field.name == "velocity") offset_v = field.offset;
    }
    
    if (offset_x < 0 || offset_y < 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Missing x/y fields");
      return;
    }
    
    // Log once
    static bool logged = false;
    if (!logged) {
      RCLCPP_INFO(get_logger(), "PointCloud fields: x@%d, y@%d, z@%d, RCS@%d (type=%d), v@%d",
                  offset_x, offset_y, offset_z, offset_rcs, rcs_datatype, offset_v);
      logged = true;
    }
    
    // Reset per-frame counters
    {
      std::lock_guard<std::mutex> lock(grid_mutex_);
      for (auto &cell : cells_) {
        cell.detection_count_this_frame = 0;
      }
    }
    
    // Process points
    size_t point_count = msg->data.size() / msg->point_step;
    int valid_points = 0;
    
    std::vector<std::pair<double, double>> hits_in_map;  // For free-space ray casting
    
    for (size_t i = 0; i < point_count; ++i) {
      const uint8_t *ptr = &msg->data[i * msg->point_step];
      
      float x_radar = *reinterpret_cast<const float*>(ptr + offset_x);
      float y_radar = *reinterpret_cast<const float*>(ptr + offset_y);
      float z_radar = offset_z >= 0 ? *reinterpret_cast<const float*>(ptr + offset_z) : 0.0f;
      
      if (!std::isfinite(x_radar) || !std::isfinite(y_radar)) continue;
      
      float range = std::sqrt(x_radar * x_radar + y_radar * y_radar);
      if (range < params_.min_range || range > params_.max_range) continue;
      
      // Read RCS
      float rcs = 128.0f;  // Default
      if (offset_rcs >= 0) {
        if (rcs_datatype == sensor_msgs::msg::PointField::UINT8) {
          rcs = static_cast<float>(*(ptr + offset_rcs));
        } else if (rcs_datatype == sensor_msgs::msg::PointField::FLOAT32) {
          rcs = *reinterpret_cast<const float*>(ptr + offset_rcs);
        }
      }
      
      // Read velocity
      float velocity = 0.0f;
      if (offset_v >= 0) {
        velocity = *reinterpret_cast<const float*>(ptr + offset_v);
      }
      
      // Transform point to map frame
      double x_map, y_map, z_map;
      if (!transform_point_(x_radar, y_radar, z_radar, radar_to_map, x_map, y_map, z_map)) {
        continue;
      }
      
      // Update grid cell
      int gx, gy;
      if (world_to_grid_(x_map, y_map, gx, gy)) {
        int idx = grid_index_(gx, gy);
        update_cell_occupied_(idx, rcs, velocity);
        hits_in_map.push_back({x_map, y_map});
        valid_points++;
      }
    }
    
    // Mark free space along rays from robot to each hit
    if (have_robot_pose_ && params_.mark_free_space) {
      for (const auto& hit : hits_in_map) {
        mark_free_space_ray_(robot_x_in_map_, robot_y_in_map_, hit.first, hit.second);
      }
    }
    
    // Publish
    publish_grid_();
    
    RCLCPP_DEBUG(get_logger(), "Frame %d: %d points in map frame", current_frame_, valid_points);
  }

  void publish_grid_() {
    std::lock_guard<std::mutex> lock(grid_mutex_);
    
    grid_msg_.header.stamp = this->now();
    
    for (size_t i = 0; i < cells_.size(); ++i) {
      const CellState &cell = cells_[i];
      
      // Check if we should publish this cell
      if (cell.confidence < params_.confidence_publish_threshold && 
          cell.observation_count < 2) {
        grid_msg_.data[i] = -1;  // Unknown
        continue;
      }
      
      if (std::abs(cell.log_odds) < 0.05f) {
        grid_msg_.data[i] = -1;  // Unknown
        continue;
      }
      
      float prob = probability_from_logodds_(cell.log_odds);
      int8_t occ = static_cast<int8_t>(std::clamp<int>(
        static_cast<int>(std::round(prob * 100.0f)), 0, 100));
      
      grid_msg_.data[i] = occ;
    }
    
    pub_grid_->publish(grid_msg_);
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<IndoorRadarOccupancyGrid>();
  
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
  executor.add_node(node);
  
  RCLCPP_INFO(node->get_logger(), "Starting Radar Occupancy Grid v2");
  executor.spin();
  
  rclcpp::shutdown();
  return 0;
}