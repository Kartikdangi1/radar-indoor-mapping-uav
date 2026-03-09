#include "fused_odometry/radar_scan_matcher.hpp"
#include <cmath>
#include <algorithm>

namespace fused_odometry {

RadarScanMatcher::RadarScanMatcher(const Config& cfg)
    : cfg_(cfg)
{
    // Configure GICP
    gicp_.setMaxCorrespondenceDistance(cfg_.max_correspondence_distance);
    gicp_.setMaximumIterations(cfg_.max_iterations);
    gicp_.setTransformationEpsilon(cfg_.transformation_epsilon);
    gicp_.setEuclideanFitnessEpsilon(cfg_.euclidean_fitness_epsilon);
    // Use 5 neighbors for covariance estimation (default 20 crashes on sparse radar clouds)
    gicp_.setCorrespondenceRandomness(5);
}

Eigen::Matrix4f RadarScanMatcher::buildInitialGuess(const Eigen::Vector3d& v_ego_body,
                                                     const Eigen::Quaterniond& R_imu_delta,
                                                     double dt) const
{
    // Predict position change from velocity
    Eigen::Vector3f delta_pos = (v_ego_body * dt).cast<float>();

    // Use IMU rotation as initial orientation guess
    Eigen::Matrix3f R = R_imu_delta.cast<float>().toRotationMatrix();

    // Build 4×4 transformation matrix
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = delta_pos;

    return T;
}

void RadarScanMatcher::estimatePoseCovariance(double fitness,
                                               size_t inlier_count,
                                               size_t total_points,
                                               Eigen::Matrix<double, 6, 6>& covariance) const
{
    // Inlier ratio: fraction of points that were matched
    double inlier_ratio = (total_points > 0) ? (double)inlier_count / total_points : 0.0;

    // Scale uncertainty inversely with fitness and inlier ratio
    // High fitness (→1) and high inlier ratio → low uncertainty
    // Low fitness and low inlier ratio → high uncertainty
    double fitness_scale = 1.0 - std::clamp(fitness, 0.0, 1.0);  // [0, 1]
    double inlier_scale = 1.0 - std::clamp(inlier_ratio, 0.0, 1.0);  // [0, 1]

    // Combined scaling factor
    double scale = 1.0 + fitness_scale * 2.0 + inlier_scale * 2.0;  // [1, 5]

    covariance = Eigen::Matrix<double, 6, 6>::Zero();

    // Position covariance (diagonal, in m²)
    double pos_cov = cfg_.base_position_uncertainty * scale;
    covariance(0, 0) = pos_cov;  // x
    covariance(1, 1) = pos_cov;  // y
    covariance(2, 2) = pos_cov * 2.0;  // z (typically less observable from radar)

    // Rotation covariance (diagonal, in rad²)
    double rot_cov = cfg_.base_rotation_uncertainty * scale;
    covariance(3, 3) = rot_cov * 2.0;  // roll (weak from horizontal radar)
    covariance(4, 4) = rot_cov * 2.0;  // pitch (weak from horizontal radar)
    covariance(5, 5) = rot_cov;  // yaw (strong from scan matching)
}

bool RadarScanMatcher::match(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& prev,
                              const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& curr,
                              const Eigen::Vector3d& v_ego_body,
                              const Eigen::Quaterniond& R_imu_delta,
                              double dt,
                              Eigen::Vector3d& delta_pos,
                              Eigen::Quaterniond& delta_rot,
                              Eigen::Matrix<double, 6, 6>& pose_covariance)
{
    // Sanity checks
    if (!prev || !curr || prev->empty() || curr->empty()) {
        return false;
    }

    size_t prev_size = prev->size();
    size_t curr_size = curr->size();

    // Need enough points for GICP (must exceed setCorrespondenceRandomness, which is 5)
    if (curr_size < 8 || prev_size < 8) {
        return false;
    }

    // Build initial transformation guess
    Eigen::Matrix4f T_init = buildInitialGuess(v_ego_body, R_imu_delta, dt);

    // Configure and run GICP
    gicp_.setInputSource(curr);  // Source = current scan
    gicp_.setInputTarget(prev);  // Target = previous scan
    pcl::PointCloud<pcl::PointXYZ> aligned;
    gicp_.align(aligned, T_init);

    // Check if convergence succeeded
    if (!gicp_.hasConverged()) {
        last_fitness_ = 1.0;  // worst case
        last_inlier_count_ = 0;
        return false;
    }

    // Extract result
    Eigen::Matrix4f T_result = gicp_.getFinalTransformation();
    double fitness = gicp_.getFitnessScore();

    // Count inliers (points with correspondence within max_correspondence_distance)
    // Approximate: assume all matched points in aligned cloud are inliers
    last_fitness_ = fitness;
    last_inlier_count_ = aligned.size();

    // Check inlier ratio
    double inlier_ratio = (double)last_inlier_count_ / curr_size;
    if (inlier_ratio < cfg_.min_inlier_ratio) {
        return false;  // Too few matches
    }

    // Extract position and rotation from result
    delta_pos = T_result.block<3, 1>(0, 3).cast<double>();
    Eigen::Matrix3f R_mat = T_result.block<3, 3>(0, 0);
    delta_rot = Eigen::Quaterniond(R_mat.cast<double>()).normalized();

    // Estimate covariance based on match quality
    estimatePoseCovariance(fitness, last_inlier_count_, curr_size, pose_covariance);

    return true;
}

}  // namespace fused_odometry
