#include "fused_odometry/lidar_processor.hpp"

namespace fused_odometry {

void LidarProcessor::processRange(const sensor_msgs::msg::Range::SharedPtr msg,
                                  double roll_rad, double pitch_rad)
{
    double range = static_cast<double>(msg->range);
    double min_r = static_cast<double>(msg->min_range);
    double max_r = static_cast<double>(msg->max_range);

    // Apply config overrides if set
    if (config_.min_range > 0) min_r = config_.min_range;
    if (config_.max_range > 0) max_r = config_.max_range;

    if (!std::isfinite(range) || range < min_r || range > max_r) {
        valid_ = false;
        return;
    }

    // Correct for tilt: height = range * cos(roll) * cos(pitch)
    double z_raw = range * std::cos(roll_rad) * std::cos(pitch_rad);

    // Apply flip if configured
    if (config_.flip_z) z_raw = -z_raw;

    // EMA filter
    if (std::isfinite(height_)) {
        height_ = config_.ema_alpha * z_raw + (1.0 - config_.ema_alpha) * height_;
    } else {
        height_ = z_raw;
    }

    valid_ = true;
}

}  // namespace fused_odometry
