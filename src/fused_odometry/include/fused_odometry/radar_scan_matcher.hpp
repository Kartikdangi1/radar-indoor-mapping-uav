#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <memory>

namespace fused_odometry {

/**
 * Radar Scan Matching: ICP-based registration of consecutive radar point clouds.
 *
 * Input:
 *   - prev_cloud: filtered radar inliers from previous scan
 *   - curr_cloud: filtered radar inliers from current scan
 *   - v_ego: estimated ego-velocity in body frame [m/s]
 *   - R_imu_delta: change in IMU orientation since last scan
 *
 * Output:
 *   - delta_pos: relative position change [m]
 *   - delta_rot: relative rotation change (quaternion)
 *   - covariance: 6×6 pose uncertainty (for factor graph weighting)
 */
class RadarScanMatcher {
public:
    struct Config {
        // GICP parameters
        double max_correspondence_distance = 1.5;  // [m] correspondence threshold
        int max_iterations = 50;                   // max ICP iterations
        double transformation_epsilon = 1e-4;      // convergence threshold
        double euclidean_fitness_epsilon = 1e-5;   // fitness threshold

        // Covariance estimation
        double base_position_uncertainty = 0.1;    // [m²] base variance
        double base_rotation_uncertainty = 0.01;   // [rad²] base variance
        double min_inlier_ratio = 0.3;             // require ≥30% points matched
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit RadarScanMatcher(const Config& cfg);
    ~RadarScanMatcher() = default;

    /**
     * Match two consecutive radar scans.
     *
     * @param prev Previous scan (filtered inliers from /fused_odom/radar_inliers)
     * @param curr Current scan (filtered inliers from /fused_odom/radar_inliers)
     * @param v_ego_body Ego-velocity in body frame [m/s] (for initial guess)
     * @param R_imu_delta Change in IMU orientation [quaternion] (for initial guess)
     * @param dt Time elapsed since last scan [s] (for integrating v_ego)
     * @param[out] delta_pos Relative position change [m]
     * @param[out] delta_rot Relative rotation as quaternion
     * @param[out] pose_covariance 6×6 covariance: [cov_x, cov_y, cov_z, cov_roll, cov_pitch, cov_yaw]
     *
     * @return true if match succeeded with sufficient inlier ratio
     */
    bool match(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& prev,
               const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& curr,
               const Eigen::Vector3d& v_ego_body,
               const Eigen::Quaterniond& R_imu_delta,
               double dt,
               Eigen::Vector3d& delta_pos,
               Eigen::Quaterniond& delta_rot,
               Eigen::Matrix<double, 6, 6>& pose_covariance);

    /**
     * Returns fitness score from last match (0–1, higher is better).
     * Useful for adaptive weighting in factor graph.
     */
    double getLastFitness() const { return last_fitness_; }

    /**
     * Returns number of inliers from last match.
     */
    size_t getLastInlierCount() const { return last_inlier_count_; }

private:
    Config cfg_;
    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp_;

    // Cached results from last match
    double last_fitness_{0.0};
    size_t last_inlier_count_{0};

    /**
     * Build initial transform guess from velocity + IMU rotation.
     * Initial guess significantly improves ICP convergence.
     */
    Eigen::Matrix4f buildInitialGuess(const Eigen::Vector3d& v_ego_body,
                                       const Eigen::Quaterniond& R_imu_delta,
                                       double dt) const;

    /**
     * Estimate pose covariance based on ICP fitness and match statistics.
     * Higher uncertainty when fewer points matched or fitness is low.
     */
    void estimatePoseCovariance(double fitness,
                                size_t inlier_count,
                                size_t total_points,
                                Eigen::Matrix<double, 6, 6>& covariance) const;
};

}  // namespace fused_odometry
