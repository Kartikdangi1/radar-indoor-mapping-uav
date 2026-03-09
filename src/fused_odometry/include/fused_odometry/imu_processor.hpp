#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sensor_msgs/msg/imu.hpp>
#include <rclcpp/rclcpp.hpp>
#include <cmath>
#include <string>

namespace fused_odometry {

/**
 * Madgwick AHRS filter for 6-DOF orientation estimation (no magnetometer).
 * Roll/pitch converge via gravity; yaw drifts (gyro-integrated only).
 */
class MadgwickAHRS {
public:
    explicit MadgwickAHRS(double beta = 0.033) : beta_(beta) {}

    void setBeta(double b) { beta_ = b; }
    bool isInitialized() const { return initialized_; }

    void update(double gx, double gy, double gz,
                double ax, double ay, double az, double dt);

    void getQuaternion(double& w, double& x, double& y, double& z) const {
        w = q0_; x = q1_; y = q2_; z = q3_;
    }

    void getEuler(double& roll, double& pitch, double& yaw) const;

private:
    void integrate(double d1, double d2, double d3, double d4, double dt);

    double beta_;
    double q0_{1.0}, q1_{0.0}, q2_{0.0}, q3_{0.0};
    bool   initialized_{false};
};

/**
 * IMU Processor: runs Madgwick filter and applies extrinsic calibration.
 */
class IMUProcessor {
public:
    struct Config {
        bool enabled = true;
        std::string topic = "/robot/sensor/imu/data";
        double madgwick_beta = 0.033;
        Eigen::Matrix3d R_imu_to_base = Eigen::Matrix3d::Identity();
        bool flip_x = false;  // Flip X axis (multiply by -1)
        bool flip_y = false;  // Flip Y axis (multiply by -1)
        bool flip_z = false;  // Flip Z axis (multiply by -1)
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit IMUProcessor(const Config& cfg) : config_(cfg), madgwick_(cfg.madgwick_beta) {}

    void processIMU(const sensor_msgs::msg::Imu::SharedPtr msg, double dt = 0.0);

    // Output: calibrated quaternion and raw accel/gyro
    void getOutput(Eigen::Quaterniond& q_base, double& ax, double& ay, double& az,
                   double& gx, double& gy, double& gz) const {
        q_base = q_base_;
        ax = ax_; ay = ay_; az = az_;
        gx = gx_; gy = gy_; gz = gz_;
    }

    // Also return raw Madgwick quat for imu_pose topic
    void getRawQuaternion(double& qw, double& qx, double& qy, double& qz) const {
        madgwick_.getQuaternion(qw, qx, qy, qz);
    }

    bool isReady() const { return madgwick_.isInitialized(); }

private:
    Config config_;
    MadgwickAHRS madgwick_;
    Eigen::Quaterniond q_base_{Eigen::Quaterniond::Identity()};
    double ax_{0}, ay_{0}, az_{0};
    double gx_{0}, gy_{0}, gz_{0};
    rclcpp::Time prev_time_;
    bool time_valid_{false};
};

}  // namespace fused_odometry
