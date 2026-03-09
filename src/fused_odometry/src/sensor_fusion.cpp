#include "fused_odometry/sensor_fusion.hpp"
#include <cmath>
#include <algorithm>

namespace fused_odometry {

// ============================================================================
// Initialization
// ============================================================================

void SensorFusion::initializeHome() {
    if (home_set_) return;
    pos_home_ = pos_;
    att_home_ = att_;
    home_set_ = true;
}

// ============================================================================
// Quaternion / Lie Group Helpers
// ============================================================================

Eigen::Matrix3d SensorFusion::quatToRotMat(const Eigen::Quaterniond& q) const {
    return q.normalized().toRotationMatrix();
}

// Exponential map: rotation vector θ (3D) → quaternion
// θ encodes axis * angle; quaternion uses half-angle internally.
Eigen::Quaterniond SensorFusion::expMap(const Eigen::Vector3d& theta) const {
    double angle = theta.norm();
    if (angle < 1e-8) {
        return Eigen::Quaterniond::Identity();
    }
    Eigen::Vector3d axis = theta / angle;
    double half = angle * 0.5;
    return Eigen::Quaterniond(std::cos(half), axis.x()*std::sin(half),
                              axis.y()*std::sin(half), axis.z()*std::sin(half));
}

// Logarithm map: quaternion → rotation vector (3D)
Eigen::Vector3d SensorFusion::logMap(const Eigen::Quaterniond& q) const {
    Eigen::Quaterniond qn = q.normalized();
    if (std::abs(qn.w()) >= 1.0) return Eigen::Vector3d::Zero();
    double norm = std::sqrt(qn.x()*qn.x() + qn.y()*qn.y() + qn.z()*qn.z());
    if (norm < 1e-8) return Eigen::Vector3d::Zero();
    double angle = 2.0 * std::atan2(norm, qn.w());
    return (angle / norm) * Eigen::Vector3d(qn.x(), qn.y(), qn.z());
}

// ============================================================================
// IMU Propagation (at 480 Hz)
// ============================================================================

void SensorFusion::propagateIMU(double ax, double ay, double az,
                                double gx, double gy, double gz, double dt)
{
    if (dt <= 0.0 || dt > 1.0) return;

    if (imu_count_ < config_.initialization_imu_frames) {
        imu_count_++;
        return;  // Still initializing
    }

    if (!ekf_initialized_) {
        initializeHome();
        ekf_initialized_ = true;
    }

    // Apply bias correction
    Eigen::Vector3d acc_meas(ax, ay, az);
    Eigen::Vector3d gyro_meas(gx, gy, gz);
    Eigen::Vector3d acc_corrected = acc_meas - b_acc_;
    Eigen::Vector3d gyro_corrected = gyro_meas - b_gyro_;

    // Attitude from body to world
    Eigen::Matrix3d R = quatToRotMat(att_);

    // Rotate acceleration to world frame
    Eigen::Vector3d acc_world = R * acc_corrected;

    // Subtract gravity: IMU measures specific force (accel + gravity)
    Eigen::Vector3d acc_free = acc_world - g_vec_;

    // 1. Update position: p += v*dt + 0.5*a*dt²
    pos_ += vel_ * dt + 0.5 * acc_free * (dt * dt);

    // 2. Update velocity: v += a*dt
    vel_ += acc_free * dt;

    // 3. Update attitude: q = q ⊗ exp(ω * dt)
    // Using quaternion exponential for rotation integration
    Eigen::Quaterniond dq = expMap(gyro_corrected * dt);
    att_ = (att_ * dq).normalized();

    // 4. Propagate covariance P
    // F ≈ [I  I*dt  0   0       0     ]
    //     [0  I     -R  0       0     ]
    //     [0  0     I   0       -R*dt ]
    //     [0  0     0   I       0     ]
    //     [0  0     0   0       I     ]
    // Simplified: P = F*P*F^T + Q

    // Construct F (15×15)
    // State: [δp(0:2), δv(3:5), δθ(6:8), δb_a(9:11), δb_g(12:14)]
    Eigen::Matrix<double, 15, 15> F = Eigen::Matrix<double, 15, 15>::Identity();
    F.block<3,3>(0, 3) = Eigen::Matrix3d::Identity() * dt;       // dp/dv: p += v*dt

    // Skew-symmetric matrix of rotated acceleration for attitude Jacobian
    Eigen::Matrix3d acc_skew;
    Eigen::Vector3d Ra = R * acc_corrected;
    acc_skew <<       0, -Ra.z(),  Ra.y(),
                Ra.z(),       0, -Ra.x(),
               -Ra.y(),  Ra.x(),       0;
    F.block<3,3>(3, 6) = -acc_skew * dt;                         // dv/dθ: velocity w.r.t. attitude error
    F.block<3,3>(3, 9) = -R * dt;                                // dv/db_a: velocity w.r.t. accel bias
    F.block<3,3>(6, 12) = -Eigen::Matrix3d::Identity() * dt;     // dθ/db_g: attitude w.r.t. gyro bias

    // Process noise Q (15×15)
    Eigen::Matrix<double, 15, 15> Q = Eigen::Matrix<double, 15, 15>::Zero();
    double qa = config_.acc_noise * dt;
    double qg = config_.gyro_noise * dt;
    double qba = config_.acc_bias_noise * dt;
    double qbg = config_.gyro_bias_noise * dt;

    Q.block<3,3>(3, 3)   = Eigen::Matrix3d::Identity() * (qa*qa);
    Q.block<3,3>(6, 6)   = Eigen::Matrix3d::Identity() * (qg*qg);
    Q.block<3,3>(9, 9)   = Eigen::Matrix3d::Identity() * (qba*qba);
    Q.block<3,3>(12, 12) = Eigen::Matrix3d::Identity() * (qbg*qbg);

    // P = F*P*F^T + Q
    P_ = F * P_ * F.transpose() + Q;

    // Enforce symmetry and bound covariance
    P_ = (P_ + P_.transpose()) * 0.5;
    P_.diagonal() = P_.diagonal().cwiseMin(10.0);  // Cap diagonal elements
}

// ============================================================================
// Radar Velocity Update
// ============================================================================

void SensorFusion::updateRadarVelocity(const Eigen::Vector3d& v_body_meas,
                                        const Eigen::Vector3d& v_body_sigma,
                                        int /*imu_count*/)
{
    if (!ekf_initialized_) return;

    // Measurement: velocity in body frame
    // Convert to odom frame for comparison with EKF state
    Eigen::Matrix3d R = quatToRotMat(att_);
    Eigen::Vector3d vel_body_pred = R.transpose() * vel_;

    // Innovation: difference between measured and predicted velocity
    Eigen::Vector3d y = v_body_meas - vel_body_pred;

    // Measurement noise R (3×3)
    // Use per-axis sigma from RANSAC, scaled by config
    Eigen::Matrix3d R_meas = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 3; ++i) {
        double s = std::max(v_body_sigma(i), 0.05);
        R_meas(i, i) = s * s * config_.r_vel_scale;
    }

    // Measurement matrix H (3×15)
    // Maps state to body-frame velocity: z = R^T * v
    Eigen::Matrix<double, 3, 15> H = Eigen::Matrix<double, 3, 15>::Zero();
    H.block<3,3>(0, 3) = R.transpose();  // Velocity is in state[3:6]

    // Innovation covariance: S = H*P*H^T + R
    Eigen::Matrix3d S = H * P_ * H.transpose() + R_meas;

    // Kalman gain: K = P*H^T*S^{-1}
    Eigen::Matrix<double, 15, 3> K = P_ * H.transpose() * S.inverse();

    // State update: x += K*y
    Eigen::Vector<double, 15> dx = K * y;
    pos_ += dx.segment<3>(0);
    vel_ += dx.segment<3>(3);
    Eigen::Vector3d dtheta = dx.segment<3>(6);
    att_ = (att_ * expMap(dtheta)).normalized();
    b_acc_ += dx.segment<3>(9);
    b_gyro_ += dx.segment<3>(12);

    // Covariance update: P = (I - K*H)*P
    Eigen::Matrix<double, 15, 15> I_KH = Eigen::Matrix<double, 15, 15>::Identity() - K * H;
    P_ = I_KH * P_;
    P_ = (P_ + P_.transpose()) * 0.5;  // Enforce symmetry
}

// ============================================================================
// LiDAR Height Update
// ============================================================================

void SensorFusion::updateLidarHeight(double z_meas, bool valid)
{
    if (!ekf_initialized_ || !valid) return;

    // Measurement: height (z in odom frame)
    double z_pred = pos_.z();
    double innov = z_meas - z_pred;

    // Measurement matrix H (1×15): only pos.z
    Eigen::Matrix<double, 1, 15> H = Eigen::Matrix<double, 1, 15>::Zero();
    H(0, 2) = 1.0;

    // Innovation covariance: S = H*P*H^T + R
    double S = (H * P_ * H.transpose())(0, 0) + config_.r_height;

    // Kalman gain: K = P*H^T / S
    Eigen::Matrix<double, 15, 1> K = P_ * H.transpose() / S;

    // State update: x += K*innov
    Eigen::Vector<double, 15> dx = K * innov;
    pos_ += dx.segment<3>(0);
    vel_ += dx.segment<3>(3);
    Eigen::Vector3d dtheta = dx.segment<3>(6);
    att_ = (att_ * expMap(dtheta)).normalized();
    b_acc_ += dx.segment<3>(9);
    b_gyro_ += dx.segment<3>(12);

    // Covariance update
    Eigen::Matrix<double, 15, 15> I_KH = Eigen::Matrix<double, 15, 15>::Identity() - K * H;
    P_ = I_KH * P_;
    P_ = (P_ + P_.transpose()) * 0.5;
}

// ============================================================================
// Legacy Interface (backward compatible)
// ============================================================================

bool SensorFusion::fuse(Eigen::Quaterniond& q_fused, double& z_fused,
                        double& pos_x, double& pos_y, double& pos_cov_x, double& pos_cov_y,
                        Eigen::Quaterniond& q_imu_raw)
{
    if (!ekf_initialized_) return false;

    // Return raw IMU for /fused_odom/imu_pose topic
    q_imu_raw = q_imu_raw_;

    // Apply radar and lidar updates if they came in
    if (radar_valid_ && imu_count_ >= config_.initialization_imu_frames) {
        updateRadarVelocity(radar_vel_, radar_sigma_, imu_count_);
    }

    if (lidar_valid_ && imu_count_ >= config_.initialization_imu_frames) {
        updateLidarHeight(lidar_height_, true);
    }

    // Express relative to home
    Eigen::Quaterniond q_rel = (att_home_.inverse() * att_).normalized();
    Eigen::Vector3d pos_rel = pos_ - pos_home_;

    q_fused = q_rel;
    pos_x = pos_rel.x();
    pos_y = pos_rel.y();
    z_fused = pos_rel.z();

    // Covariance from P
    pos_cov_x = std::min(P_(0, 0), 100.0);
    pos_cov_y = std::min(P_(1, 1), 100.0);

    return true;
}

}  // namespace fused_odometry
