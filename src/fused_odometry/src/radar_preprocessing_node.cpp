#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "fused_odometry/radar_processor.hpp"
#include <cmath>
#include <string>
#include <algorithm>
#include <memory>

namespace fused_odometry {

class RadarPreprocessingNode : public rclcpp::Node {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RadarPreprocessingNode() : Node("radar_preprocessing_node")
    {
        // ── Topics ───────────────────────────────────────────────────────
        input_topic_       = declare_parameter("input_topic",  std::string("/PointCloudDetection"));
        output_topic_      = declare_parameter("output_topic", std::string("/PointCloudDetectionFiltered"));

        // ── Range filtering ──────────────────────────────────────────────
        min_range_         = declare_parameter("min_range",   0.5);
        max_range_         = declare_parameter("max_range",  30.0);

        // ── Field-of-view filtering ──────────────────────────────────────
        max_azimuth_deg_   = declare_parameter("max_azimuth_deg",   60.0);
        max_elevation_deg_ = declare_parameter("max_elevation_deg", 30.0);

        // ── RCS filtering ────────────────────────────────────────────────
        min_rcs_threshold_ = declare_parameter("min_rcs_threshold", -20.0);

        // ── Coordinate frame flips ───────────────────────────────────────
        flip_x_            = declare_parameter("flip_x",  true);   // Invert X: fixes real-left→virtual-right
        flip_y_            = declare_parameter("flip_y",  false);
        flip_z_            = declare_parameter("flip_z",  true);

        // ── Z-slice filtering (disabled by default) ──────────────────────
        enable_z_slice_    = declare_parameter("enable_z_slice", false);
        z_min_             = declare_parameter("z_min", -0.5);
        z_max_             = declare_parameter("z_max",  1.5);

        // ── TF frame transformation (optional) ───────────────────────────
        enable_tf_transform_   = declare_parameter("enable_tf_transform", false);
        target_frame_          = declare_parameter("target_frame", std::string("base_link"));
        tf_lookup_timeout_ms_  = static_cast<int>(
            declare_parameter("tf_lookup_timeout_sec", 0.1) * 1000.0);

        // ── Debug logging ────────────────────────────────────────────────
        log_filter_stats_  = declare_parameter("log_filter_stats", true);

        // Convert degrees to radians
        az_lim_rad_ = max_azimuth_deg_   * M_PI / 180.0;
        el_lim_rad_ = max_elevation_deg_ * M_PI / 180.0;

        // ── TF setup (if enabled) ────────────────────────────────────────
        if (enable_tf_transform_) {
            tf_buffer_   = std::make_unique<tf2_ros::Buffer>(get_clock());
            tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
            RCLCPP_INFO(get_logger(), "TF transform enabled: will transform points to '%s'", target_frame_.c_str());
        }

        // ── Publishers and Subscribers ───────────────────────────────────
        pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, rclcpp::SensorDataQoS());
        sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic_, rclcpp::SensorDataQoS(),
            std::bind(&RadarPreprocessingNode::cloud_callback, this, std::placeholders::_1));

        // ── Startup summary ──────────────────────────────────────────────
        RCLCPP_INFO(get_logger(), "RadarPreprocessingNode ready:");
        RCLCPP_INFO(get_logger(), "  %s → %s", input_topic_.c_str(), output_topic_.c_str());
        RCLCPP_INFO(get_logger(),
            "  Range [%.1f, %.1f] m  |  Az ±%.0f°  El ±%.0f°  |  RCS ≥ %.0f",
            min_range_, max_range_, max_azimuth_deg_, max_elevation_deg_, min_rcs_threshold_);
        RCLCPP_INFO(get_logger(),
            "  Z-slice: %s [%.2f, %.2f] m  |  TF: %s",
            enable_z_slice_ ? "ON" : "OFF", z_min_, z_max_,
            enable_tf_transform_ ? "ON" : "OFF");
    }

private:
    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<RadarPointCloudType> raw_cloud;
        pcl::fromROSMsg(*msg, raw_cloud);

        const int8_t rcs_thresh = static_cast<int8_t>(std::clamp(
            static_cast<int>(min_rcs_threshold_), -128, 127));
        pcl::PointCloud<RadarPointCloudType> filtered_cloud;
        filtered_cloud.reserve(raw_cloud.size());

        // ── Apply all filtering stages ───────────────────────────────────
        for (auto p : raw_cloud) {
            // Sanity check: finite coordinates
            if (!std::isfinite(p.x) || !std::isfinite(p.y) ||
                !std::isfinite(p.z) || !std::isfinite(p.v)) continue;

            // ── Range filtering ──────────────────────────────────────────
            const double r = std::sqrt(
                static_cast<double>(p.x)*p.x + static_cast<double>(p.y)*p.y +
                static_cast<double>(p.z)*p.z);
            if (r < min_range_ || r > max_range_) continue;

            // ── Azimuth FoV filtering ────────────────────────────────────
            const double az = std::atan2(static_cast<double>(p.y), static_cast<double>(p.x));
            if (std::abs(az) > az_lim_rad_) continue;

            // ── Elevation FoV filtering (rejects floor/ceiling clutter) ──
            const double el = std::asin(std::clamp(static_cast<double>(p.z) / r, -1.0, 1.0));
            if (std::abs(el) > el_lim_rad_) continue;

            // ── RCS filtering (stricter indoors to reject weak clutter) ──
            if (p.RCS < rcs_thresh) continue;

            // ── Coordinate frame flips (if mounted inverted) ──────────────
            if (flip_x_) p.x = -p.x;
            if (flip_y_) p.y = -p.y;
            if (flip_z_) p.z = -p.z;

            // ── Z-slice filtering (aggressively reject floor/ceiling) ────
            if (enable_z_slice_) {
                if (static_cast<double>(p.z) < z_min_ || static_cast<double>(p.z) > z_max_) continue;
            }

            filtered_cloud.push_back(p);
        }

        // ── Optional: Transform points to target frame via TF ────────────
        if (enable_tf_transform_ && !filtered_cloud.empty()) {
            filtered_cloud = transformCloud(filtered_cloud, msg->header.frame_id);
        }

        // ── Publish filtered cloud ───────────────────────────────────────
        sensor_msgs::msg::PointCloud2 out_msg;
        pcl::toROSMsg(filtered_cloud, out_msg);
        out_msg.header = msg->header;
        if (enable_tf_transform_) {
            out_msg.header.frame_id = target_frame_;
        }
        pub_->publish(out_msg);

        // ── Log filter statistics ────────────────────────────────────────
        if (log_filter_stats_) {
            const double rejection_rate = raw_cloud.empty()
                ? 0.0
                : (1.0 - static_cast<double>(filtered_cloud.size()) / raw_cloud.size()) * 100.0;
            RCLCPP_DEBUG(get_logger(),
                "Filter: %zu → %zu points (%.1f%% rejected)",
                raw_cloud.size(), filtered_cloud.size(), rejection_rate);
        }
    }

    /**
     * Transform point cloud to target frame using TF2.
     * Preserves all custom fields (v, RCS).
     */
    pcl::PointCloud<RadarPointCloudType> transformCloud(
        const pcl::PointCloud<RadarPointCloudType>& cloud_in,
        const std::string& source_frame)
    {
        pcl::PointCloud<RadarPointCloudType> cloud_out = cloud_in;

        if (!tf_buffer_ || source_frame == target_frame_) {
            return cloud_out;
        }

        try {
            // Lookup transform with timeout
            const auto transform = tf_buffer_->lookupTransform(
                target_frame_, source_frame,
                tf2::TimePointZero,  // latest available transform
                tf2::durationFromSec(tf_lookup_timeout_ms_ / 1000.0));

            // Extract rotation and translation
            tf2::Transform tf2_transform;
            tf2::fromMsg(transform.transform, tf2_transform);

            // Apply transform to each point
            for (auto& p : cloud_out) {
                tf2::Vector3 v(p.x, p.y, p.z);
                tf2::Vector3 v_transformed = tf2_transform * v;
                p.x = static_cast<float>(v_transformed.x());
                p.y = static_cast<float>(v_transformed.y());
                p.z = static_cast<float>(v_transformed.z());
                // Note: Doppler velocity (p.v) is preserved as-is;
                // rotation would require quaternion multiplication which is complex
            }

            RCLCPP_DEBUG(get_logger(), "Transformed %zu points from '%s' to '%s'",
                cloud_out.size(), source_frame.c_str(), target_frame_.c_str());

        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN(get_logger(),
                "TF lookup failed (%s → %s): %s. Keeping original frame.",
                source_frame.c_str(), target_frame_.c_str(), ex.what());
        }

        return cloud_out;
    }

    std::string input_topic_, output_topic_;
    double  min_range_, max_range_, max_azimuth_deg_, max_elevation_deg_;
    double  min_rcs_threshold_;
    bool    flip_x_, flip_y_, flip_z_, enable_z_slice_;
    double  z_min_, z_max_, az_lim_rad_, el_lim_rad_;
    bool    enable_tf_transform_, log_filter_stats_;
    std::string target_frame_;
    int     tf_lookup_timeout_ms_;

    // TF2 buffer and listener (optional)
    std::unique_ptr<tf2_ros::Buffer>           tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    pub_;
};

}  // namespace fused_odometry

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<fused_odometry::RadarPreprocessingNode>());
    rclcpp::shutdown();
    return 0;
}
