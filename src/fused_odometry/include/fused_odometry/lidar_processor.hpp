#pragma once

#include <sensor_msgs/msg/range.hpp>
#include <cmath>
#include <limits>
#include <string>

namespace fused_odometry {

/**
 * LiDAR Height Processor: EMA filter + tilt correction for altitude measurement.
 */
class LidarProcessor {
public:
    struct Config {
        bool enabled = true;
        std::string topic = "/robot/sensor/lidar/downwards/data";
        double ema_alpha = 0.15;
        bool flip_z = true;
        double min_range = 0.05;
        double max_range = 5.0;
    };

    explicit LidarProcessor(const Config& cfg) : config_(cfg) {}

    void processRange(const sensor_msgs::msg::Range::SharedPtr msg,
                     double roll_rad, double pitch_rad);

    bool isValid() const { return valid_; }
    double getHeight() const { return height_; }

private:
    Config config_;
    double height_{std::numeric_limits<double>::quiet_NaN()};
    bool   valid_{false};
};

}  // namespace fused_odometry
