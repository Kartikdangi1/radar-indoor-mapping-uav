#include "temporal_radar_mapping/temporal_radar_occupancy_grid.hpp"

namespace radar_mapping {

bool IndoorRadarOccupancyGrid::get_transform_(const std::string& target_frame, const std::string& source_frame,
                                              const rclcpp::Time& time, geometry_msgs::msg::TransformStamped& transform) const {
  // Strategy: try stamped lookup first, then latest available.
  // Use short timeout to avoid blocking during bag time jumps.
  try {
    transform = tf_buffer_->lookupTransform(
        target_frame, source_frame, time,
        rclcpp::Duration::from_seconds(std::min(params_.tf_timeout_sec, 0.1)));
    return true;
  } catch (tf2::TransformException&) {
    try {
      // Fallback: latest available transform (survives time jumps)
      transform = tf_buffer_->lookupTransform(target_frame, source_frame, tf2::TimePointZero);
      return true;
    } catch (tf2::TransformException&) {
      // Suppress warning — caller will handle the failure with better context
      return false;
    }
  }
}


bool IndoorRadarOccupancyGrid::transform_point_(double x_in, double y_in, double z_in,
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


bool IndoorRadarOccupancyGrid::is_occluded_by_map_(double x_start, double y_start,
                                                    double x_end, double y_end, double range_end) const {
  if (!params_.enable_map_occlusion || current_frame_ < params_.map_occlusion_start_frame) {
    return false;
  }

  int sx, sy, ex, ey;
  if (!world_to_grid_(x_start, y_start, sx, sy)) return false;
  if (!world_to_grid_(x_end, y_end, ex, ey)) return false;

  std::vector<std::pair<int,int>> ray_cells;
  bresenham_(sx, sy, ex, ey, ray_cells);

  for (size_t i = 0; i < ray_cells.size() - 1; ++i) {
    int gx = ray_cells[i].first;
    int gy = ray_cells[i].second;

    double cell_x = grid_origin_x_ + (gx + 0.5) * params_.grid_resolution;
    double cell_y = grid_origin_y_ + (gy + 0.5) * params_.grid_resolution;
    double cell_range = std::sqrt((cell_x - x_start) * (cell_x - x_start) +
                                  (cell_y - y_start) * (cell_y - y_start));

    int idx = grid_index_(gx, gy);
    const CellState& cell = cells_[idx];

    if (cell.is_static_confirmed ||
        (cell.occupied_count >= params_.min_hits_to_occlude &&
         cell.log_odds > params_.occupied_protection_threshold)) {
      if (range_end > cell_range + params_.behind_range_margin_m) {
        return true;
      }
    }
  }

  return false;
}


void IndoorRadarOccupancyGrid::reject_multipath_(std::vector<RadarDetection>& detections) const {
  if (!params_.enable_multipath_rejection || detections.empty()) return;

  std::unordered_map<int, std::vector<size_t>> azimuth_bins;
  for (size_t i = 0; i < detections.size(); ++i) {
    int bin = static_cast<int>(std::round(detections[i].azimuth * 180.0 / M_PI /
                                          params_.azimuth_bin_size_deg));
    azimuth_bins[bin].push_back(i);
  }

  for (auto& [bin_id, indices] : azimuth_bins) {
    if (static_cast<int>(indices.size()) < params_.multipath_min_near_hits + 1) continue;

    // Sort by range
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
      return detections[a].range < detections[b].range;
    });

    // Find median range of nearest cluster
    int near_count = std::min(static_cast<int>(indices.size()),
                               params_.multipath_min_near_hits);
    float near_median_range = detections[indices[near_count / 2]].range;

    // Mark far points as multipath ghosts (occluded)
    float multipath_threshold = near_median_range * static_cast<float>(params_.multipath_range_ratio);
    for (size_t idx : indices) {
      if (detections[idx].range > multipath_threshold) {
        detections[idx].is_occluded = true;
      }
    }
  }
}

}  // namespace radar_mapping
