#include "temporal_radar_mapping/temporal_radar_occupancy_grid.hpp"

namespace radar_mapping {

float IndoorRadarOccupancyGrid::get_min_rcs_u8_for_angle_(float azimuth_rad, float elevation_rad) const {
  float az_deg = std::abs(azimuth_rad) * 180.0f / M_PI;
  float el_deg = std::abs(elevation_rad) * 180.0f / M_PI;

  if (el_deg > 10.0f) {
    return static_cast<float>(params_.min_rcs_u8_elevation);
  }

  if (az_deg < 20.0f) {
    return static_cast<float>(params_.min_rcs_u8_center);
  } else if (az_deg < 40.0f) {
    return static_cast<float>(params_.min_rcs_u8_edge);
  } else {
    return static_cast<float>(params_.min_rcs_u8_high_edge);
  }
}


float IndoorRadarOccupancyGrid::normalize_rcs_(float raw_value) const {
  if (params_.rcs_format == "uint8") {
    float range = static_cast<float>(params_.rcs_max_db - params_.rcs_min_db);
    return (raw_value / 255.0f) * range + static_cast<float>(params_.rcs_min_db);
  }
  return raw_value;
}


float IndoorRadarOccupancyGrid::intensity_to_occupancy_(float raw_intensity, float range_m) const {
  float rcs_db = normalize_rcs_(raw_intensity);

  if (rcs_db < params_.min_rcs_threshold_db) {
    return static_cast<float>(params_.p0);
  }

  float normalized = (rcs_db - static_cast<float>(params_.rcs_min_db)) /
                     static_cast<float>(params_.rcs_max_db - params_.rcs_min_db);
  normalized = std::clamp(normalized, 0.0f, 1.0f);

  float range_factor = std::exp(-params_.range_decay_factor * range_m);
  float rcs_contrib = normalized * range_factor;
  float prob = static_cast<float>(params_.p_occ) * (1.0f - static_cast<float>(params_.intensity_scale)) +
               (0.55f + rcs_contrib * 0.40f) * static_cast<float>(params_.intensity_scale);

  return std::clamp(prob, 0.5f, 0.95f);
}


void IndoorRadarOccupancyGrid::bresenham_(int x0, int y0, int x1, int y1, std::vector<std::pair<int,int>>& cells) const {
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


float IndoorRadarOccupancyGrid::compensate_radial_velocity_(float v_measured, double x_radar, double y_radar) const {
  if (!params_.enable_ego_compensation || !have_robot_velocity_) {
    return v_measured;
  }

  double range = std::sqrt(x_radar * x_radar + y_radar * y_radar);
  if (range < 0.01) return v_measured;

  double ray_x = x_radar / range;
  double ray_y = y_radar / range;
  float v_radial_ego = static_cast<float>(robot_vx_ * ray_x + robot_vy_ * ray_y);

  return v_measured - v_radial_ego;
}


void IndoorRadarOccupancyGrid::apply_gaussian_occupied_update_(
    double hit_x, double hit_y,
    float base_lo_inc,
    const rclcpp::Time& stamp,
    bool is_occluded,
    float rcs_u8,
    float velocity_compensated)
{
  int cx, cy;
  if (!world_to_grid_(hit_x, hit_y, cx, cy)) return;

  // ── Center cell: full state update ──
  {
    int idx = grid_index_(cx, cy);
    CellState& cell = cells_[idx];

    bool first_hit = (cell.last_updated_frame != current_frame_);
    if (first_hit) {
      cell.observation_count++;
      if (!is_occluded) cell.occupied_count++;
      cell.last_updated_frame = current_frame_;
    }

    cell.is_behind_detection = is_occluded;

    if (params_.enable_dynamic_filter) {
      cell.velocity_accumulator = 0.9f * cell.velocity_accumulator +
                                   0.1f * std::abs(velocity_compensated);
      if (cell.velocity_accumulator > params_.dynamic_velocity_threshold) {
        cell.is_dynamic = true;
        cell.dynamic_observations++;
      } else if (cell.dynamic_observations > 0) {
        cell.dynamic_observations--;
        if (cell.dynamic_observations == 0) cell.is_dynamic = false;
      }
    }
    if (cell.is_static_confirmed &&
        cell.dynamic_observations > params_.dynamic_revoke_threshold) {
      cell.is_static_confirmed = false;
    }

    float rcs_db = normalize_rcs_(rcs_u8);
    cell.avg_rcs = 0.8f * cell.avg_rcs + 0.2f * rcs_db;

    float lo_inc = base_lo_inc;
    if (is_occluded) lo_inc *= static_cast<float>(params_.behind_detection_weight);
    cell.log_odds += lo_inc;
    cell.log_odds = std::clamp(cell.log_odds,
                               static_cast<float>(params_.lo_min),
                               static_cast<float>(params_.lo_max));

    cell.confidence = std::min(1.0f, cell.confidence +
                               static_cast<float>(params_.confidence_increment));
    cell.last_seen_time = stamp;
    if (!is_occluded) cell.last_occupied_time = stamp;

    if (cell.occupied_count >= params_.min_observations_for_static && !cell.is_dynamic)
      cell.is_static_confirmed = true;
  }

  // ── Tangential Gaussian spread — only if enabled, not occluded, and center confident ──
  if (!params_.enable_gaussian_sensor_model || is_occluded) return;
  int center_occupied = cells_[grid_index_(cx, cy)].occupied_count;
  if (center_occupied < params_.gaussian_min_hits) return;

  // Robot→detection ray direction
  double dx_world = hit_x - robot_x_in_map_;
  double dy_world = hit_y - robot_y_in_map_;
  double range = std::sqrt(dx_world * dx_world + dy_world * dy_world);
  if (range < 0.01) return;
  double radial_x = dx_world / range;
  double radial_y = dy_world / range;
  // Tangential unit vector (perpendicular to ray, in-plane)
  double tang_x = -radial_y;
  double tang_y =  radial_x;

  const double sigma = params_.gaussian_sigma_m;
  const double two_sigma_sq = 2.0 * sigma * sigma;
  const int r_half = params_.gaussian_radial_half;
  const int t_half = params_.gaussian_tangential_half;
  const int loop_half = std::max(r_half, t_half);

  for (int dy = -loop_half; dy <= loop_half; ++dy) {
    for (int dx = -loop_half; dx <= loop_half; ++dx) {
      if (dx == 0 && dy == 0) continue;  // center already handled

      int gx = cx + dx;
      int gy = cy + dy;
      if (gx < 0 || gx >= grid_width_ || gy < 0 || gy >= grid_height_) continue;

      // World-frame offset from detection center
      double off_x = dx * params_.grid_resolution;
      double off_y = dy * params_.grid_resolution;

      // Decompose into radial and tangential components
      double d_radial     = off_x * radial_x + off_y * radial_y;
      double d_tangential = off_x * tang_x   + off_y * tang_y;

      // Reject cells outside radial tolerance
      if (std::abs(d_radial) > r_half * params_.grid_resolution + 1e-6) continue;

      // Tangential Gaussian weight
      float weight = static_cast<float>(
          std::exp(-d_tangential * d_tangential / two_sigma_sq));
      if (weight < 0.05f) continue;

      int idx = grid_index_(gx, gy);
      CellState& cell = cells_[idx];

      float lo_inc = base_lo_inc * weight;
      cell.log_odds += lo_inc;
      cell.log_odds = std::clamp(cell.log_odds,
                                 static_cast<float>(params_.lo_min),
                                 static_cast<float>(params_.lo_max));
      cell.last_seen_time = stamp;
      // Note: occupied_count, last_occupied_time, confidence NOT updated
      // → free-space rays can always clear these cells (occupied_count stays 0)
    }
  }
}


void IndoorRadarOccupancyGrid::mark_free_space_ray_(double robot_x, double robot_y, double hit_x, double hit_y,
                                                    const rclcpp::Time& stamp) {
  if (!params_.mark_free_space) return;

  int sx, sy, ex, ey;
  if (!world_to_grid_(robot_x, robot_y, sx, sy)) return;
  if (!world_to_grid_(hit_x, hit_y, ex, ey)) return;

  double hit_range = std::sqrt((hit_x - robot_x) * (hit_x - robot_x) +
                               (hit_y - robot_y) * (hit_y - robot_y));

  std::vector<std::pair<int, int>> ray_cells;
  bresenham_(sx, sy, ex, ey, ray_cells);
  if (ray_cells.empty()) return;

  const float lo_free_inc = logit_(static_cast<float>(params_.p_free)) -
                             logit_(static_cast<float>(params_.p0));
  const float weighted_free = lo_free_inc * static_cast<float>(params_.free_space_weight);

  for (size_t i = 0; i < ray_cells.size() - 1; ++i) {
    int gx = ray_cells[i].first;
    int gy = ray_cells[i].second;

    double cell_x = grid_origin_x_ + (gx + 0.5) * params_.grid_resolution;
    double cell_y = grid_origin_y_ + (gy + 0.5) * params_.grid_resolution;
    double cell_range = std::sqrt((cell_x - robot_x) * (cell_x - robot_x) +
                                  (cell_y - robot_y) * (cell_y - robot_y));

    if (cell_range >= hit_range - params_.safety_margin_m - params_.grid_resolution / 2.0) {
      break;
    }

    int idx = grid_index_(gx, gy);
    CellState& cell = cells_[idx];

    // Only protect cells directly observed enough times (not Gaussian-spread neighbors)
    if (cell.occupied_count >= params_.gaussian_min_hits) continue;

    if (cell.last_occupied_time.nanoseconds() > 0) {
      double age = (stamp - cell.last_occupied_time).seconds();
      if (age < params_.occupied_protection_sec) continue;
    }

    if (cell.is_static_confirmed) continue;

    cell.log_odds += weighted_free;
    cell.log_odds = std::max(cell.log_odds, static_cast<float>(params_.lo_min));
    cell.last_seen_time = stamp;
    cell.observation_count++;
  }
}

}  // namespace radar_mapping
