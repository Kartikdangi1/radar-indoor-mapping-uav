#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <algorithm>

namespace fused_odometry {

/**
 * Multi-sensor fusion: 9-state IMU-driven EKF combining IMU, LiDAR, and Radar.
 *
 * State: [p(3), v(3), err_att(3), b_acc(3), b_gyro(3)] = 15 elements in error-state form
 * - p: position in odom frame [m]
 * - v: velocity in odom frame [m/s]
 * - err_att: error-state attitude (3D perturbation, exp-mapped to quaternion)
 * - b_acc: accelerometer bias [m/s²]
 * - b_gyro: gyroscope bias [rad/s]
 *
 * Propagation: at IMU rate (~480 Hz) using raw accel+gyro - bias
 * Updates: Radar Doppler on velocity; LiDAR on height
 */
class SensorFusion {
public:
    struct Config {
        bool use_imu = true;
        bool use_lidar = true;
        bool use_radar = true;
        bool use_2d_mode = true;
        bool is_holonomic = false;
        bool publish_tf = false;
        int initialization_imu_frames = 50;
        Eigen::Matrix3d R_imu_to_base = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d R_radar_to_base = Eigen::Matrix3d::Identity();

        // ── IMU Propagation Noise ────────────────────────────────────────
        // Process noise per second
        double acc_noise = 0.05;        // [m/s²] accel random walk
        double gyro_noise = 0.005;      // [rad/s] gyro random walk
        double acc_bias_noise = 0.001;  // [m/s³] accel bias random walk
        double gyro_bias_noise = 0.0001; // [rad/s²] gyro bias random walk

        // ── Measurement Noise ────────────────────────────────────────────
        // Radar velocity measurement noise (scaled by RANSAC sigma)
        double r_vel_scale = 1.0;       // multiplier on RANSAC sigma²
        // LiDAR height measurement noise
        double r_height = 0.01;         // [m²] variance
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit SensorFusion(const Config& cfg) : config_(cfg),
        g_vec_(0, 0, 9.81) {}

    // ── IMU Propagation (called at IMU rate ~480 Hz) ────────────────────
    void propagateIMU(double ax, double ay, double az,
                      double gx, double gy, double gz, double dt);

    // ── Measurement Updates ───────────────────────────────────────────────
    void updateRadarVelocity(const Eigen::Vector3d& v_body_meas,
                            const Eigen::Vector3d& v_body_sigma,
                            int imu_count);

    void updateLidarHeight(double z_meas, bool valid);

    // ── Legacy setters (for backward compatibility) ──────────────────────
    void setIMUData(const Eigen::Quaterniond& q_imu, double ax, double ay, double az,
                    double gx, double gy, double gz, int imu_count) {
        q_imu_raw_ = q_imu;
        ax_ = ax; ay_ = ay; az_ = az;
        gx_ = gx; gy_ = gy; gz_ = gz;
        imu_count_ = imu_count;
    }
    void setLiDARData(bool valid, double height) {
        lidar_valid_ = valid;
        lidar_height_ = height;
    }
    void setRadarData(bool valid, double dt, const Eigen::Vector3d& vel, const Eigen::Vector3d& sigma) {
        radar_valid_ = valid;
        radar_dt_ = dt;
        radar_vel_ = vel;
        radar_sigma_ = sigma;
    }

    // ── Legacy fuse (still calls new update methods, deprecated) ──────────
    bool fuse(Eigen::Quaterniond& q_fused, double& z_fused,
              double& pos_x, double& pos_y, double& pos_cov_x, double& pos_cov_y,
              Eigen::Quaterniond& q_imu_raw);

    // Accessors for publishing
    double getAccelX() const { return ax_; }
    double getAccelY() const { return ay_; }
    double getAccelZ() const { return az_; }
    double getGyroX() const { return gx_; }
    double getGyroY() const { return gy_; }
    double getGyroZ() const { return gz_; }

private:
    Config config_;
    Eigen::Vector3d g_vec_;  // gravity vector [0, 0, 9.81]

    // ── 9-state EKF State ─────────────────────────────────────────────────
    // P(3): position in odom frame [m]
    // V(3): velocity in odom frame [m/s]
    // Q(4): attitude quaternion (nominal)
    // b_a(3): accelerometer bias [m/s²]
    // b_g(3): gyroscope bias [rad/s]
    Eigen::Vector3d pos_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d vel_{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond att_{Eigen::Quaterniond::Identity()};
    Eigen::Vector3d b_acc_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d b_gyro_{Eigen::Vector3d::Zero()};

    // ── Error-State Covariance (15×15) ───────────────────────────────────
    // Error states: δp, δv, δθ, δb_a, δb_g (15 DOF)
    Eigen::Matrix<double, 15, 15> P_{Eigen::Matrix<double, 15, 15>::Identity() * 0.1};
    bool ekf_initialized_{false};

    // ── Home/Reference Frame ──────────────────────────────────────────────
    bool home_set_{false};
    Eigen::Vector3d pos_home_{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond att_home_{Eigen::Quaterniond::Identity()};

    // ── Sensor data (stored for accessors) ──────────────────────────────
    Eigen::Quaterniond q_imu_raw_{Eigen::Quaterniond::Identity()};
    double ax_{0}, ay_{0}, az_{0};
    double gx_{0}, gy_{0}, gz_{0};
    int imu_count_{0};

    bool lidar_valid_{false};
    double lidar_height_{0.0};

    bool radar_valid_{false};
    double radar_dt_{0.0};
    Eigen::Vector3d radar_vel_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d radar_sigma_{Eigen::Vector3d(9999, 9999, 9999)};

    // ── Helper methods ──────────────────────────────────────────────────
    void initializeHome();
    Eigen::Matrix3d quatToRotMat(const Eigen::Quaterniond& q) const;
    Eigen::Quaterniond expMap(const Eigen::Vector3d& theta) const;
    Eigen::Vector3d logMap(const Eigen::Quaterniond& q) const;
};

}  // namespace fused_odometry
