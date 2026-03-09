#include "temporal_radar_mapping/temporal_radar_occupancy_grid.hpp"

namespace radar_mapping {

void IndoorRadarOccupancyGrid::init_grid_() {
  grid_width_ = static_cast<int>(std::ceil(params_.grid_size_m / params_.grid_resolution));
  grid_height_ = grid_width_;
  const size_t num_cells = static_cast<size_t>(grid_width_) * static_cast<size_t>(grid_height_);
  cells_.resize(num_cells);

  rclcpp::Time init_time(0, 0, RCL_ROS_TIME);
  for (auto &cell : cells_) {
    cell.last_seen_time = init_time;
    cell.last_occupied_time = init_time;
  }

  grid_origin_x_ = -params_.grid_size_m / 2.0;
  grid_origin_y_ = -params_.grid_size_m / 2.0;

  grid_msg_.info.resolution = static_cast<float>(params_.grid_resolution);
  grid_msg_.info.width = static_cast<uint32_t>(grid_width_);
  grid_msg_.info.height = static_cast<uint32_t>(grid_height_);
  grid_msg_.info.origin.position.x = grid_origin_x_;
  grid_msg_.info.origin.position.y = grid_origin_y_;
  grid_msg_.info.origin.position.z = 0.0;
  grid_msg_.info.origin.orientation.w = 1.0;
  grid_msg_.header.frame_id = params_.map_frame;
  grid_msg_.data.resize(num_cells, -1);
}


void IndoorRadarOccupancyGrid::maybe_recenter_grid_() {
  if (!params_.enable_moving_origin || !have_robot_pose_) return;

  // Robot position in grid cells
  int robot_gx = static_cast<int>(std::floor((robot_x_in_map_ - grid_origin_x_) / params_.grid_resolution));
  int robot_gy = static_cast<int>(std::floor((robot_y_in_map_ - grid_origin_y_) / params_.grid_resolution));

  int margin = static_cast<int>(std::ceil(params_.recenter_margin_m / params_.grid_resolution));
  int step   = static_cast<int>(std::ceil(params_.expand_step_m   / params_.grid_resolution));

  // Determine how many cells to add on each side
  int expand_left   = (robot_gx < margin)                    ? step + (margin - robot_gx) : 0;
  int expand_right  = (robot_gx >= grid_width_  - margin)    ? step + (robot_gx - (grid_width_  - margin) + 1) : 0;
  int expand_bottom = (robot_gy < margin)                    ? step + (margin - robot_gy) : 0;
  int expand_top    = (robot_gy >= grid_height_ - margin)    ? step + (robot_gy - (grid_height_ - margin) + 1) : 0;

  if (expand_left == 0 && expand_right == 0 && expand_bottom == 0 && expand_top == 0) return;

  int new_width  = grid_width_  + expand_left + expand_right;
  int new_height = grid_height_ + expand_bottom + expand_top;
  double new_origin_x = grid_origin_x_ - expand_left  * params_.grid_resolution;
  double new_origin_y = grid_origin_y_ - expand_bottom * params_.grid_resolution;

  RCLCPP_INFO(get_logger(),
    "Expanding grid: +L%d +R%d +B%d +T%d → %dx%d cells (%.0f×%.0f m)",
    expand_left, expand_right, expand_bottom, expand_top,
    new_width, new_height,
    new_width * params_.grid_resolution, new_height * params_.grid_resolution);

  // Allocate new grid and copy existing cells (offset by expand_left/expand_bottom)
  rclcpp::Time init_time(0, 0, RCL_ROS_TIME);
  std::vector<CellState> new_cells(static_cast<size_t>(new_width) * new_height);
  for (auto& c : new_cells) { c.last_seen_time = init_time; c.last_occupied_time = init_time; }

  for (int gy = 0; gy < grid_height_; ++gy) {
    int ny = gy + expand_bottom;
    for (int gx = 0; gx < grid_width_; ++gx) {
      int nx = gx + expand_left;
      new_cells[ny * new_width + nx] = cells_[gy * grid_width_ + gx];
    }
  }

  cells_        = std::move(new_cells);
  grid_width_   = new_width;
  grid_height_  = new_height;
  grid_origin_x_ = new_origin_x;
  grid_origin_y_ = new_origin_y;

  grid_msg_.info.width            = static_cast<uint32_t>(grid_width_);
  grid_msg_.info.height           = static_cast<uint32_t>(grid_height_);
  grid_msg_.info.origin.position.x = grid_origin_x_;
  grid_msg_.info.origin.position.y = grid_origin_y_;
  grid_msg_.data.resize(static_cast<size_t>(grid_width_) * grid_height_, -1);
}


bool IndoorRadarOccupancyGrid::world_to_grid_(double x, double y, int& gx, int& gy) const {
  const double ox = grid_origin_x_;
  const double oy = grid_origin_y_;
  const double r = params_.grid_resolution;
  gx = static_cast<int>(std::floor((x - ox) / r));
  gy = static_cast<int>(std::floor((y - oy) / r));
  return (gx >= 0 && gy >= 0 && gx < grid_width_ && gy < grid_height_);
}


int IndoorRadarOccupancyGrid::grid_index_(int gx, int gy) const {
  return gy * grid_width_ + gx;
}


float IndoorRadarOccupancyGrid::logit_(float p) const {
  p = std::clamp(p, 0.01f, 0.99f);
  return std::log(p / (1.0f - p));
}


float IndoorRadarOccupancyGrid::probability_from_logodds_(float lo) const {
  return 1.0f / (1.0f + std::exp(-lo));
}

}  // namespace radar_mapping
