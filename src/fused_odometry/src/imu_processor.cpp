#include "fused_odometry/imu_processor.hpp"
#include <cmath>

namespace fused_odometry {

void MadgwickAHRS::update(double gx, double gy, double gz,
                          double ax, double ay, double az, double dt)
{
    double qDot1 = 0.5 * (-q1_*gx - q2_*gy - q3_*gz);
    double qDot2 = 0.5 * ( q0_*gx + q2_*gz - q3_*gy);
    double qDot3 = 0.5 * ( q0_*gy - q1_*gz + q3_*gx);
    double qDot4 = 0.5 * ( q0_*gz + q1_*gy - q2_*gx);

    double aNorm = std::sqrt(ax*ax + ay*ay + az*az);
    if (aNorm > 0.1 && aNorm < 20.0) {
        double rn = 1.0 / aNorm;
        ax *= rn; ay *= rn; az *= rn;

        double q0q0=q0_*q0_, q1q1=q1_*q1_, q2q2=q2_*q2_, q3q3=q3_*q3_;
        double _2q0=2.0*q0_, _2q1=2.0*q1_, _2q2=2.0*q2_, _2q3=2.0*q3_;
        double _4q0=4.0*q0_, _4q1=4.0*q1_, _4q2=4.0*q2_;
        double _8q1=8.0*q1_, _8q2=8.0*q2_;

        double s0 = _4q0*q2q2 + _2q2*ax + _4q0*q1q1 - _2q1*ay;
        double s1 = _4q1*q3q3 - _2q3*ax + 4.0*q0q0*q1_ - _2q0*ay
                    - _4q1 + _8q1*q1q1 + _8q1*q2q2 + _4q1*az;
        double s2 = 4.0*q0q0*q2_ + _2q0*ax + _4q2*q3q3 - _2q3*ay
                    - _4q2 + _8q2*q1q1 + _8q2*q2q2 + _4q2*az;
        double s3 = 4.0*q1q1*q3_ - _2q1*ax + 4.0*q2q2*q3_ - _2q2*ay;

        double sn = 1.0 / std::sqrt(s0*s0 + s1*s1 + s2*s2 + s3*s3);
        s0 *= sn; s1 *= sn; s2 *= sn; s3 *= sn;

        qDot1 -= beta_*s0; qDot2 -= beta_*s1;
        qDot3 -= beta_*s2; qDot4 -= beta_*s3;
    }
    integrate(qDot1, qDot2, qDot3, qDot4, dt);
}

void MadgwickAHRS::integrate(double d1, double d2, double d3, double d4, double dt) {
    q0_ += d1*dt; q1_ += d2*dt; q2_ += d3*dt; q3_ += d4*dt;
    double n = 1.0 / std::sqrt(q0_*q0_+q1_*q1_+q2_*q2_+q3_*q3_);
    q0_ *= n; q1_ *= n; q2_ *= n; q3_ *= n;
    initialized_ = true;
}

void MadgwickAHRS::getEuler(double& roll, double& pitch, double& yaw) const {
    roll  = std::atan2(2.0*(q0_*q1_+q2_*q3_), 1.0-2.0*(q1_*q1_+q2_*q2_));
    double sinp = 2.0*(q0_*q2_-q3_*q1_);
    pitch = std::abs(sinp) >= 1.0 ? std::copysign(M_PI/2.0, sinp) : std::asin(sinp);
    yaw   = std::atan2(2.0*(q0_*q3_+q1_*q2_), 1.0-2.0*(q2_*q2_+q3_*q3_));
}

// ============================================================================
// IMUProcessor
// ============================================================================

void IMUProcessor::processIMU(const sensor_msgs::msg::Imu::SharedPtr msg, double dt_override)
{
    double ax = msg->linear_acceleration.x;
    double ay = msg->linear_acceleration.y;
    double az = msg->linear_acceleration.z;
    double gx = msg->angular_velocity.x;
    double gy = msg->angular_velocity.y;
    double gz = msg->angular_velocity.z;

    // Apply axis flips if configured
    if (config_.flip_x) { ax = -ax; gx = -gx; }
    if (config_.flip_y) { ay = -ay; gy = -gy; }
    if (config_.flip_z) { az = -az; gz = -gz; }

    double dt = dt_override;
    if (!dt_override && time_valid_) {
        double d = (rclcpp::Time(msg->header.stamp) - prev_time_).seconds();
        if (d > 0.0 && d < 0.5) dt = d;
    }
    prev_time_ = rclcpp::Time(msg->header.stamp);
    time_valid_ = true;

    madgwick_.update(gx, gy, gz, ax, ay, az, dt);

    // Store raw accel/gyro (after flips)
    ax_ = ax; ay_ = ay; az_ = az;
    gx_ = gx; gy_ = gy; gz_ = gz;

    // Apply extrinsic calibration to get base_link orientation
    double qw, qx, qy, qz;
    madgwick_.getQuaternion(qw, qx, qy, qz);
    const Eigen::Quaterniond q_imu(qw, qx, qy, qz);
    q_base_ = (q_imu * Eigen::Quaterniond(config_.R_imu_to_base)).normalized();
}

}  // namespace fused_odometry
