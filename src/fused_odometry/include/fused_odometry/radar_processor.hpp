#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <random>
#include <vector>
#include <string>

namespace fused_odometry {

/**
 * Custom PCL point type with Doppler velocity.
 * Matches /PointCloudDetection fields: x, y, z, v (Doppler), RCS
 */
struct RadarPointCloudType {
    PCL_ADD_POINT4D;
    float v;      // Doppler velocity
    int8_t RCS;   // RCS as intensity
    PCL_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * Radar Ego-Velocity Processor: RANSAC + Doppler velocity estimation.
 */
class RadarProcessor {
public:
    struct Config {
        bool enabled = true;
        std::string topic = "/PointCloudDetection";
        double min_dist = 0.5;
        double max_dist = 180.0;
        double min_db = 5.0;
        double azimuth_thresh_deg = 60.0;
        double elevation_thresh_deg = 30.0;
        double inlier_thresh = 0.15;
        double doppler_velocity_correction_factor = 1.0;
        unsigned N_ransac_points = 8;
        double success_prob = 0.99;
        double outlier_prob = 0.40;
        double dynamic_removal_thresh = 0.0;  // Dynamic object removal [m/s].
                                               // Points whose Doppler residual
                                               // (|measured - predicted_from_v_prev|)
                                               // exceeds this are classified as dynamic
                                               // and excluded before RANSAC.
                                               // Set to 0.0 to disable.
        bool flip_x = false;  // Flip X velocity (multiply by -1)
        bool flip_y = false;  // Flip Y velocity (multiply by -1)
        bool flip_z = false;  // Flip Z velocity (multiply by -1)

        // When true, rotate radar points by IMU orientation before RANSAC.
        // RANSAC then directly outputs world-frame velocity, eliminating the
        // need for q_pose * v_body rotation in dead-reckoning.
        bool points_in_world_frame = false;

        // Adaptive EMA smoothing (lateral observability fix)
        // When inlier count is low (lateral motion → poor Doppler observability),
        // trust the previous smoothed estimate more to avoid "stuck→jump" artefacts.
        // Disable (enable_ema = false) when using a factor graph — the FG already
        // provides optimal smoothing; EMA introduces lag and hides uncertainty.
        bool enable_ema = true;
        double alpha_high_inliers = 0.5;    // alpha when inliers >= inlier_count_thresh
        double alpha_low_inliers  = 0.15;   // alpha when inliers <  inlier_count_thresh
        unsigned inlier_count_thresh = 20;  // threshold for switching alpha
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit RadarProcessor(const Config& cfg);

    bool estimate(const sensor_msgs::msg::PointCloud2& cloud_msg,
                  double pitch, double roll, double yaw, bool holonomic,
                  Eigen::Vector3d& v_r, Eigen::Vector3d& sigma,
                  sensor_msgs::msg::PointCloud2& inliers_msg,
                  sensor_msgs::msg::PointCloud2& outliers_msg,
                  sensor_msgs::msg::PointCloud2& dynamic_msg,
                  const Eigen::Quaterniond& q_world = Eigen::Quaterniond::Identity());

private:
    struct RadarEgoVelState {
        Eigen::Vector3d best_v = Eigen::Vector3d::Zero();
        std::vector<size_t> best_inliers;
    };

    void ransac(const std::vector<RadarPointCloudType>& pts,
                const Eigen::MatrixXd& H, const Eigen::VectorXd& d,
                double pitch, double roll, bool holonomic,
                RadarEgoVelState& state);

    Eigen::Vector3d solveNonHolonomic(const Eigen::MatrixXd& H,
                                      const Eigen::VectorXd& d) const;

    void buildInlierOutlierMsgs(
        const std::vector<RadarPointCloudType>& pts,
        const std::vector<size_t>& inlier_idx,
        const std_msgs::msg::Header& header,
        sensor_msgs::msg::PointCloud2& in_msg,
        sensor_msgs::msg::PointCloud2& out_msg);

    Config config_;
    std::mt19937 rng_;
    unsigned ransac_iterations_{100};
    Eigen::Vector3d v_smoothed_{Eigen::Vector3d::Zero()};
    bool v_smoothed_init_{false};
    // Previous frame ego-velocity (in pts frame, pre-flip) for dynamic filter
    Eigen::Vector3d v_prev_{Eigen::Vector3d::Zero()};
    bool            v_prev_valid_{false};

    void updateRansacIterations() {
        double p = std::pow(1.0 - config_.outlier_prob, config_.N_ransac_points);
        if (p <= 0.0 || p >= 1.0) { ransac_iterations_ = 500; return; }
        ransac_iterations_ = static_cast<unsigned>(
            std::ceil(std::log(1.0 - config_.success_prob) / std::log(1.0 - p)));
        ransac_iterations_ = std::min(ransac_iterations_, 500u);
        ransac_iterations_ = std::max(ransac_iterations_, 10u);
    }
};

}  // namespace fused_odometry

// Register custom point type with PCL
POINT_CLOUD_REGISTER_POINT_STRUCT(fused_odometry::RadarPointCloudType,
    (float, x, x)(float, y, y)(float, z, z)
    (float, v, v)(int8_t, RCS, RCS))
