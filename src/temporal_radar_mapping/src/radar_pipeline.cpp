#include "temporal_radar_mapping/temporal_radar_occupancy_grid.hpp"

namespace radar_mapping {

void IndoorRadarOccupancyGrid::radar_callback_(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
  current_frame_++;
  rclcpp::Time msg_time = msg->header.stamp;

  std::string source_frame = msg->header.frame_id;
  if (source_frame.empty()) source_frame = params_.radar_frame;

  // V5: Get radar->base_link static transform (use TimePointZero — immune to time jumps)
  if (!have_radar_to_base_) {
    try {
      radar_to_base_cached_ = tf_buffer_->lookupTransform(
          params_.base_frame, source_frame, tf2::TimePointZero);
      have_radar_to_base_ = true;
      RCLCPP_INFO(get_logger(), "Got static radar->base_link transform");
    } catch (tf2::TransformException& ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
          "Waiting for static TF %s->%s: %s",
          source_frame.c_str(), params_.base_frame.c_str(), ex.what());
      return;
    }
  }

  // V5: Try to use fused odometry for registration (preferred over TF)
  bool use_odom_registration = false;
  OdomPose scan_odom;
  {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    if (have_fused_odom_ && have_radar_to_base_) {
      if (interpolate_odom_(msg_time, scan_odom)) {
        use_odom_registration = true;
      }
    }
  }

  // Fallback: use TF tree for registration (v4 behavior)
  geometry_msgs::msg::TransformStamped radar_to_map;
  if (!use_odom_registration) {
    // During bag playback, TF buffer may be disrupted by time jumps.
    // Try with message time first, then latest available.
    bool got_tf = get_transform_(params_.map_frame, source_frame, msg_time, radar_to_map);

    // Fallback: compute from odometry + cached static transform
    if (!got_tf && have_fused_odom_ && have_radar_to_base_) {
      std::lock_guard<std::mutex> lock(odom_mutex_);
      OdomPose fallback_odom;
      if (interpolate_odom_(msg_time, fallback_odom)) {
        // Compute: odom → base_link (from odom) + base_link → radar (cached static, inverted)
        // radar_to_map = odom_to_radar = odom_to_base_link @ base_link_to_radar
        tf2::Vector3 pos(fallback_odom.x, fallback_odom.y, 0.0);
        tf2::Quaternion quat(fallback_odom.qx, fallback_odom.qy, fallback_odom.qz, fallback_odom.qw);
        tf2::Transform odom_pose_tf2(quat, pos);

        tf2::Transform radar_to_base_tf2;
        tf2::fromMsg(radar_to_base_cached_.transform, radar_to_base_tf2);
        tf2::Transform base_to_radar_tf2 = radar_to_base_tf2.inverse();

        tf2::Transform odom_to_radar = odom_pose_tf2 * base_to_radar_tf2;

        radar_to_map.header.frame_id = params_.map_frame;
        radar_to_map.child_frame_id = source_frame;
        radar_to_map.header.stamp = msg->header.stamp;
        radar_to_map.transform = tf2::toMsg(odom_to_radar);
        got_tf = true;
      }
    }

    if (!got_tf) {
      if (!have_fused_odom_) {
        // No odom yet — this is expected during warmup
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
            "Waiting for fused odometry on %s (CREOS warmup) — %d scans skipped",
            params_.odom_topic.c_str(), frames_skipped_no_pose_);
      } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
            "No valid TF or odom for scan registration — %d scans skipped",
            frames_skipped_no_pose_);
      }
      frames_skipped_no_pose_++;
      return;
    }
  }

  // Log first successful registration
  if (!logged_first_registration_) {
    RCLCPP_INFO(get_logger(), "First scan registered at frame %d via %s (skipped %d frames during warmup)",
                current_frame_, use_odom_registration ? "fused odometry" : "TF tree",
                frames_skipped_no_pose_);
    logged_first_registration_ = true;
  }

  // Update robot pose (prefer fused odom, fallback to TF)
  if (!use_odom_registration) {
    geometry_msgs::msg::TransformStamped base_to_map;
    if (get_transform_(params_.map_frame, params_.base_frame, msg_time, base_to_map)) {
      // Fallback velocity estimation via TF differentiation
      double new_x = base_to_map.transform.translation.x;
      double new_y = base_to_map.transform.translation.y;
      if (have_robot_pose_ && last_tf_time_.nanoseconds() > 0) {
        double dt = (msg_time - last_tf_time_).seconds();
        if (dt > 0.001 && dt < 1.0) {
          robot_vx_ = (new_x - robot_x_in_map_) / dt;
          robot_vy_ = (new_y - robot_y_in_map_) / dt;
          have_robot_velocity_ = true;
        }
      }
      robot_x_in_map_ = new_x;
      robot_y_in_map_ = new_y;
      last_tf_time_ = msg_time;
      have_robot_pose_ = true;
    }
  }
  // (If using odom, robot state is already updated in odom_callback_)

  // V5: Motion compensation - compute delta pose between this scan and last
  double motion_dx = 0.0, motion_dy = 0.0;
  bool apply_motion_comp = false;
  if (params_.enable_motion_compensation && have_last_scan_time_ && use_odom_registration) {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    OdomPose prev_odom;
    if (interpolate_odom_(last_scan_time_, prev_odom)) {
      double dt = (msg_time - last_scan_time_).seconds();
      if (dt > 0.001 && dt < 0.5) {
        // Motion between scans in map frame
        motion_dx = scan_odom.x - prev_odom.x;
        motion_dy = scan_odom.y - prev_odom.y;
        apply_motion_comp = true;
      }
    }
  }

  // V5: Recenter grid if robot nears edge
  {
    std::lock_guard<std::mutex> lock(grid_mutex_);
    maybe_recenter_grid_();
  }

  // Find field offsets
  int offset_x = -1, offset_y = -1, offset_z = -1, offset_rcs = -1, offset_v = -1;
  uint8_t rcs_datatype = sensor_msgs::msg::PointField::FLOAT32;

  for (const auto& field : msg->fields) {
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

  // STEP 1: Parse all detections
  std::vector<RadarDetection> detections;
  size_t point_count = msg->data.size() / msg->point_step;

  for (size_t i = 0; i < point_count; ++i) {
    const uint8_t* ptr = &msg->data[i * msg->point_step];

    float x_radar = *reinterpret_cast<const float*>(ptr + offset_x);
    float y_radar = *reinterpret_cast<const float*>(ptr + offset_y);
    float z_radar = offset_z >= 0 ? *reinterpret_cast<const float*>(ptr + offset_z) : 0.0f;

    if (!std::isfinite(x_radar) || !std::isfinite(y_radar)) continue;

    float range = std::sqrt(x_radar * x_radar + y_radar * y_radar);
    if (range < params_.min_range || range > params_.max_range) continue;

    if (z_radar < params_.z_slice_min || z_radar > params_.z_slice_max) continue;

    float azimuth = std::atan2(y_radar, x_radar);
    float elevation = std::atan2(z_radar, range);

    if (std::abs(azimuth) * 180.0f / M_PI > params_.max_azimuth_deg) continue;
    if (std::abs(elevation) * 180.0f / M_PI > params_.max_elevation_deg) continue;

    float rcs_u8 = 128.0f;
    if (offset_rcs >= 0) {
      if (rcs_datatype == sensor_msgs::msg::PointField::UINT8) {
        rcs_u8 = static_cast<float>(*(ptr + offset_rcs));
      } else if (rcs_datatype == sensor_msgs::msg::PointField::FLOAT32) {
        rcs_u8 = *reinterpret_cast<const float*>(ptr + offset_rcs);
      }
    }

    float velocity = 0.0f;
    if (offset_v >= 0) {
      velocity = *reinterpret_cast<const float*>(ptr + offset_v);
    }

    // V5: Transform point to map frame using fused odom or TF
    double x_map, y_map, z_map;
    if (use_odom_registration) {
      if (!transform_point_via_odom_(x_radar, y_radar, z_radar,
                                      scan_odom, radar_to_base_cached_,
                                      x_map, y_map, z_map)) {
        continue;
      }
    } else {
      if (!transform_point_(x_radar, y_radar, z_radar, radar_to_map, x_map, y_map, z_map)) {
        continue;
      }
    }

    // V5: Inter-scan motion compensation
    // Points were captured during the scan interval. Approximate de-smearing
    // by distributing the motion delta across the scan (linear interpolation).
    if (apply_motion_comp && point_count > 1) {
      double frac = static_cast<double>(i) / static_cast<double>(point_count - 1);
      // Points captured earlier in the scan need more correction
      double comp_dx = motion_dx * (1.0 - frac);
      double comp_dy = motion_dy * (1.0 - frac);
      x_map -= comp_dx;
      y_map -= comp_dy;
    }

    int gx, gy;
    if (!world_to_grid_(x_map, y_map, gx, gy)) continue;

    RadarDetection det;
    det.x_radar = x_radar;
    det.y_radar = y_radar;
    det.z_radar = z_radar;
    det.x_map = x_map;
    det.y_map = y_map;
    det.range = range;
    det.azimuth = azimuth;
    det.elevation = elevation;
    det.rcs_u8 = rcs_u8;
    det.velocity = velocity;
    det.grid_idx = grid_index_(gx, gy);
    det.is_occluded = false;

    detections.push_back(det);
  }

  // STEP 2: Per-ray occlusion (azimuth binning)
  if (params_.enable_per_ray_occlusion && !detections.empty()) {
    std::unordered_map<int, std::vector<size_t>> azimuth_bins;

    for (size_t i = 0; i < detections.size(); ++i) {
      int bin = static_cast<int>(std::round(detections[i].azimuth * 180.0 / M_PI /
                                            params_.azimuth_bin_size_deg));
      azimuth_bins[bin].push_back(i);
    }

    for (auto& bin_pair : azimuth_bins) {
      auto& indices = bin_pair.second;
      if (indices.size() < 2) continue;

      std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return detections[a].range < detections[b].range;
      });

      float occluder_range = -1.0f;
      for (size_t idx : indices) {
        const auto& det = detections[idx];
        float min_rcs_thresh = get_min_rcs_u8_for_angle_(det.azimuth, det.elevation);
        if (det.rcs_u8 >= min_rcs_thresh) {
          occluder_range = det.range;
          break;
        }
      }

      if (occluder_range > 0) {
        for (size_t idx : indices) {
          if (detections[idx].range > occluder_range + params_.behind_range_margin_m) {
            detections[idx].is_occluded = true;
          }
        }
      }
    }
  }

  // V5: Indoor multipath rejection (after per-ray occlusion)
  reject_multipath_(detections);

  // STEP 3: Map-based occlusion
  if (have_robot_pose_) {
    std::lock_guard<std::mutex> lock(grid_mutex_);
    for (auto& det : detections) {
      if (!det.is_occluded) {
        if (is_occluded_by_map_(robot_x_in_map_, robot_y_in_map_,
                                 det.x_map, det.y_map, det.range)) {
          det.is_occluded = true;
        }
      }
    }
  }

  // STEP 4 + 5: Gaussian occupied updates + free-space
  std::vector<std::pair<double, double>> hits_in_map;
  int occluded_count = 0;
  {
    std::lock_guard<std::mutex> lock(grid_mutex_);
    for (const auto& det : detections) {
      float velocity_compensated = compensate_radial_velocity_(det.velocity,
                                                                det.x_radar, det.y_radar);
      float occ_prob = intensity_to_occupancy_(det.rcs_u8, det.range);
      float base_lo_inc = logit_(occ_prob) - logit_(static_cast<float>(params_.p0));
      apply_gaussian_occupied_update_(det.x_map, det.y_map, base_lo_inc, msg_time,
                                      det.is_occluded, det.rcs_u8, velocity_compensated);
      if (!det.is_occluded) {
        hits_in_map.push_back({det.x_map, det.y_map});
      } else {
        occluded_count++;
      }
    }
    if (have_robot_pose_ && params_.mark_free_space) {
      for (const auto& hit : hits_in_map) {
        mark_free_space_ray_(robot_x_in_map_, robot_y_in_map_,
                             hit.first, hit.second, msg_time);
      }
    }
  }

  // Update scan time tracking for motion compensation
  last_scan_time_ = msg_time;
  have_last_scan_time_ = true;

  publish_grid_();
  publish_filtered_cloud_(detections, msg_time);

  RCLCPP_DEBUG(get_logger(), "Frame %d: %zu det, %d occluded (%.1f%%), odom_reg=%s, motion_comp=%s",
               current_frame_, detections.size(), occluded_count,
               detections.size() > 0 ? 100.0f * occluded_count / detections.size() : 0.0f,
               use_odom_registration ? "Y" : "N",
               apply_motion_comp ? "Y" : "N");
}

}  // namespace radar_mapping
