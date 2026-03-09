/**
 * fused_odom_node.cpp
 *
 * Radar-Inertial Odometry — Sensor Fusion Edition.
 *
 * Sensors:
 *   - IMU   → Madgwick orientation (roll/pitch/yaw)
 *   - LiDAR → height / altitude (Z position prior)
 *   - Radar → RANSAC Doppler ego-velocity
 *
 * Position estimation strategy:
 *   Pure Doppler dead-reckoning integrated with IMU orientation and LiDAR height.
 *   Smooth and stable with gradual drift over long durations.
 *
 * Outputs:
 *   /fused_odom/odometry      — position + velocity + orientation
 *   /fused_odom/pose          — pose with covariance
 *   /fused_odom/imu_pose      — raw Madgwick orientation
 *   /fused_odom/imu           — IMU with fused orientation
 *   /fused_odom/radar_inliers — RANSAC inlier point cloud
 *   /fused_odom/radar_outliers — RANSAC outlier point cloud
 *
 * Configuration: config/fused_odom_config.yaml
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/range.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "fused_odometry/imu_processor.hpp"
#include "fused_odometry/lidar_processor.hpp"
#include "fused_odometry/radar_processor.hpp"
#include "fused_odometry/radar_scan_matcher.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <memory>
#include <mutex>

using namespace fused_odometry;

class FusedOdomNode : public rclcpp::Node {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FusedOdomNode() : Node("fused_odom_node") {
        loadConfig();

        // ── Publishers ────────────────────────────────────────────────────
        imu_pose_pub_    = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/fused_odom/imu_pose", 50);
        fused_imu_pub_   = create_publisher<sensor_msgs::msg::Imu>(
            "/fused_odom/imu", 50);
        fused_odom_pub_  = create_publisher<nav_msgs::msg::Odometry>(
            "/fused_odom/odometry", 50);
        fused_pose_pub_  = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/fused_odom/pose", 50);
        radar_inlier_pub_  = create_publisher<sensor_msgs::msg::PointCloud2>(
            "/fused_odom/radar_inliers", 10);
        radar_outlier_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
            "/fused_odom/radar_outliers", 10);
        radar_dynamic_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
            "/fused_odom/radar_dynamic_points", 10);

        if (publish_tf_) {
            tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        }

        // ── Subscriptions ─────────────────────────────────────────────────
        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, rclcpp::SensorDataQoS(),
            std::bind(&FusedOdomNode::imuCallback, this, std::placeholders::_1));

        lidar_sub_ = create_subscription<sensor_msgs::msg::Range>(
            lidar_topic_, rclcpp::SensorDataQoS(),
            std::bind(&FusedOdomNode::lidarCallback, this, std::placeholders::_1));

        radar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            radar_topic_, rclcpp::SensorDataQoS(),
            std::bind(&FusedOdomNode::radarCallback, this, std::placeholders::_1));

        // ── Publish timer ─────────────────────────────────────────────────
        timer_ = create_wall_timer(
            std::chrono::duration<double>(1.0 / publish_hz_),
            std::bind(&FusedOdomNode::publishCallback, this));

        RCLCPP_INFO(get_logger(),
            "FusedOdomNode STARTED — %.0f Hz  [Sensor Fusion + Doppler Dead-Reckoning]",
            publish_hz_);
        RCLCPP_INFO(get_logger(),
            "Topics: IMU=%s  LiDAR=%s  Radar=%s",
            imu_topic_.c_str(), lidar_topic_.c_str(), radar_topic_.c_str());
    }

private:
    // ── Configuration ─────────────────────────────────────────────────────
    void loadConfig() {
        publish_hz_   = declare_parameter("publish_hz", 50.0);
        odom_frame_   = declare_parameter("odom_frame", std::string("odom"));
        base_frame_   = declare_parameter("base_frame", std::string("base_link"));
        publish_tf_   = declare_parameter("publish_tf", false);
        use_2d_mode_  = declare_parameter("use_2d_mode", false);
        is_holonomic_ = declare_parameter("is_holonomic", false);
        gyro_calib_samples_ = static_cast<int>(
            declare_parameter("imu.gyro_calib_samples", 2400));  // ~5s @ 480 Hz

        // IMU
        imu_topic_ = declare_parameter("imu.topic",
            std::string("/robot/sensor/imu/data"));
        double imu_beta = declare_parameter("imu.madgwick_beta", 0.033);
        double imu_r    = declare_parameter("imu.extrinsic.roll", 0.0);
        double imu_p    = declare_parameter("imu.extrinsic.pitch", 0.0);
        double imu_y    = declare_parameter("imu.extrinsic.yaw", 0.0);

        IMUProcessor::Config imu_cfg;
        imu_cfg.madgwick_beta = imu_beta;
        R_imu_to_base_ =
            (Eigen::AngleAxisd(imu_y, Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(imu_p, Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(imu_r, Eigen::Vector3d::UnitX())).toRotationMatrix();
        imu_cfg.R_imu_to_base = R_imu_to_base_;
        imu_cfg.flip_x = declare_parameter("imu.extrinsic.flip_x", false);
        imu_cfg.flip_y = declare_parameter("imu.extrinsic.flip_y", false);
        imu_cfg.flip_z = declare_parameter("imu.extrinsic.flip_z", false);
        imu_proc_ = std::make_unique<IMUProcessor>(imu_cfg);

        // LiDAR
        lidar_topic_ = declare_parameter("lidar.topic",
            std::string("/robot/sensor/lidar/downwards/data"));
        LidarProcessor::Config lidar_cfg;
        lidar_cfg.ema_alpha = declare_parameter("lidar.ema_alpha", 0.15);
        lidar_cfg.flip_z    = declare_parameter("lidar.flip_z", true);
        lidar_cfg.min_range = declare_parameter("lidar.min_range", 0.05);
        lidar_cfg.max_range = declare_parameter("lidar.max_range", 5.0);
        lidar_proc_ = std::make_unique<LidarProcessor>(lidar_cfg);

        // Radar
        radar_topic_ = declare_parameter("radar.topic",
            std::string("/PointCloudDetectionFiltered"));
        double radar_r = declare_parameter("radar.extrinsic.roll", 0.0);
        double radar_p = declare_parameter("radar.extrinsic.pitch", 0.0);
        double radar_y = declare_parameter("radar.extrinsic.yaw", M_PI);
        R_radar_to_base_ =
            (Eigen::AngleAxisd(radar_y, Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(radar_p, Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(radar_r, Eigen::Vector3d::UnitX())).toRotationMatrix();

        RadarProcessor::Config radar_cfg;
        radar_cfg.min_dist              = declare_parameter("radar.ransac.min_dist", 0.5);
        radar_cfg.max_dist              = declare_parameter("radar.ransac.max_dist", 30.0);
        radar_cfg.min_db                = declare_parameter("radar.ransac.min_db", -30.0);
        radar_cfg.azimuth_thresh_deg    = declare_parameter("radar.ransac.azimuth_thresh_deg", 70.0);
        radar_cfg.elevation_thresh_deg  = declare_parameter("radar.ransac.elevation_thresh_deg", 45.0);
        radar_cfg.inlier_thresh         = declare_parameter("radar.ransac.inlier_thresh", 0.22);
        radar_cfg.doppler_velocity_correction_factor =
            declare_parameter("radar.ransac.doppler_velocity_correction_factor", 1.0);
        radar_cfg.N_ransac_points       = static_cast<unsigned>(
            declare_parameter("radar.ransac.N_ransac_points", 5));
        radar_cfg.alpha_high_inliers    = declare_parameter("radar.ransac.ema_alpha_high", 0.5);
        radar_cfg.alpha_low_inliers     = declare_parameter("radar.ransac.ema_alpha_low", 0.15);
        radar_cfg.inlier_count_thresh   = static_cast<unsigned>(
            declare_parameter("radar.ransac.ema_inlier_thresh", 20));
        radar_cfg.dynamic_removal_thresh =
            declare_parameter("radar.ransac.dynamic_removal_thresh", 0.0);
        max_radar_speed_ = declare_parameter("radar.ransac.max_speed", 2.0);

        // Doppler velocity smoothing via exponential moving average
        radar_cfg.enable_ema = true;

        // World-frame derotation: rotate radar points by IMU orientation before
        // RANSAC so it directly outputs v_world. Cleaner dead-reckoning and
        // gravity-leveled clouds for mapping.
        radar_cfg.points_in_world_frame =
            declare_parameter("radar.points_in_world_frame", false);
        points_in_world_frame_ = radar_cfg.points_in_world_frame;
        radar_proc_ = std::make_unique<RadarProcessor>(radar_cfg);

        // Bag loop reset
        enable_bag_loop_reset_       = declare_parameter("bag_loop_reset", false);
        bag_loop_reset_threshold_sec_ = declare_parameter("bag_loop_reset_threshold_sec", 1.0);

        // Scan matcher kept as optional future fallback
        RadarScanMatcher::Config scan_cfg;
        scan_cfg.max_correspondence_distance =
            declare_parameter("scan_matcher.max_dist", 1.5);
        scan_cfg.max_iterations =
            declare_parameter("scan_matcher.max_iter", 50);
        scan_matcher_ = std::make_unique<RadarScanMatcher>(scan_cfg);
    }

    // ── Callbacks ─────────────────────────────────────────────────────────
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(mtx_);
        imu_proc_->processIMU(msg);

        // ── Gyro bias calibration (stationary warm-up) ────────────────────
        // Collect raw gyro readings while the robot has not yet moved.
        // The mean angular velocity gives the gyroscope constant bias, which
        // is then used to initialise IMU preintegration so that delta_q stays
        // near identity during stationary periods.
        if (!gyro_bias_ready_) {
            Eigen::Quaterniond q_unused;
            double ax, ay, az, gx, gy, gz;
            imu_proc_->getOutput(q_unused, ax, ay, az, gx, gy, gz);
            gyro_calib_sum_ += Eigen::Vector3d(gx, gy, gz);
            ++gyro_calib_count_;
            if (gyro_calib_count_ >= gyro_calib_samples_) {
                current_b_gyro_ = gyro_calib_sum_ / static_cast<double>(gyro_calib_count_);
                gyro_bias_ready_ = true;
                RCLCPP_INFO(rclcpp::get_logger("fused_odom_node"),
                    "Gyro bias estimated from %d samples: [%.4f, %.4f, %.4f] rad/s",
                    gyro_calib_count_,
                    current_b_gyro_.x(), current_b_gyro_.y(), current_b_gyro_.z());
            }
        }
        last_imu_t_ = rclcpp::Time(msg->header.stamp);
    }

    void lidarCallback(const sensor_msgs::msg::Range::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(mtx_);
        double qw, qx, qy, qz;
        imu_proc_->getRawQuaternion(qw, qx, qy, qz);
        double roll  = std::atan2(2.0*(qw*qx + qy*qz), 1.0 - 2.0*(qx*qx + qy*qy));
        double pitch = std::asin(std::clamp(2.0*(qw*qy - qz*qx), -1.0, 1.0));
        lidar_proc_->processRange(msg, roll, pitch);
    }

    void radarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // ── RANSAC Doppler ego-velocity ───────────────────────────────────
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
            "Radar CB: received cloud with %u points", msg->width * msg->height);

        double roll, pitch, yaw;
        Eigen::Quaterniond q_imu_snap;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (!imu_proc_->isReady()) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                    "Radar CB: IMU not ready yet, skipping");
                return;
            }
            // Use calibrated orientation for roll/pitch/yaw extraction and
            // world-frame derotation so they are consistent with base_link.
            double ax_unused, ay_unused, az_unused, gx_unused, gy_unused, gz_unused;
            imu_proc_->getOutput(q_imu_snap, ax_unused, ay_unused, az_unused,
                                  gx_unused, gy_unused, gz_unused);
            double qw = q_imu_snap.w(), qx = q_imu_snap.x();
            double qy = q_imu_snap.y(), qz = q_imu_snap.z();
            roll  = std::atan2(2.0*(qw*qx + qy*qz), 1.0 - 2.0*(qx*qx + qy*qy));
            pitch = std::asin(std::clamp(2.0*(qw*qy - qz*qx), -1.0, 1.0));
            yaw   = std::atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz));
        }

        Eigen::Vector3d v_radar, sigma;
        sensor_msgs::msg::PointCloud2 inliers_msg, outliers_msg, dynamic_msg;
        bool success = radar_proc_->estimate(
            *msg, pitch, roll, yaw, is_holonomic_,
            v_radar, sigma, inliers_msg, outliers_msg, dynamic_msg,
            q_imu_snap);

        if (!success || inliers_msg.width < 3) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "Radar CB: RANSAC %s, inliers=%u (need ≥3)",
                success ? "OK but few inliers" : "FAILED", inliers_msg.width);
            if (inliers_msg.width > 0)  radar_inlier_pub_->publish(inliers_msg);
            if (outliers_msg.width > 0) radar_outlier_pub_->publish(outliers_msg);
            return;
        }

        // Reject physically implausible velocities — a moving object can dominate
        // RANSAC consensus and produce e.g. vz=3.8 m/s which integrates to a large
        // position jump. Any speed above max_radar_speed_ is impossible for this UAV.
        if (v_radar.norm() > max_radar_speed_) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "Radar CB: implausible velocity (%.2f m/s > %.2f m/s), skipping frame",
                v_radar.norm(), max_radar_speed_);
            return;
        }

        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
            "RANSAC: inliers=%u  vx=%.2f  vy=%.2f  vz=%.2f m/s",
            inliers_msg.width, v_radar.x(), v_radar.y(), v_radar.z());

        // Zero-velocity detection: treat as stationary if near-zero speed
        if (v_radar.norm() < 0.05) {
            v_radar.setZero();
            sigma << 0.03, 0.03, 0.05;
        }

        // When points_in_world_frame is true, RANSAC already outputs v_world
        // directly (derotated). No extrinsic rotation or q_pose rotation needed.
        // When false: v_radar is in radar frame → rotate to body frame.
        Eigen::Vector3d v_base;
        if (points_in_world_frame_) {
            v_base = v_radar;  // already in world frame
        } else {
            v_base = R_radar_to_base_ * v_radar;
        }

        {
            std::lock_guard<std::mutex> lk(mtx_);
            radar_vel_   = v_base;
            radar_sigma_ = sigma;
            radar_valid_ = true;
        }

        // ── Timing between radar frames ───────────────────────────────────
        const rclcpp::Time curr_t(msg->header.stamp);
        double dt = 0.0;
        if (last_radar_t_.nanoseconds() > 0) {
            dt = (curr_t - last_radar_t_).seconds();
        }
        last_radar_t_ = curr_t;

        // ── Bag loop reset (backward time jump) ──────────────────────────
        if (enable_bag_loop_reset_ && dt < -bag_loop_reset_threshold_sec_) {
            RCLCPP_WARN(get_logger(),
                "Bag loop detected (dt=%.2f s) — resetting pose to origin", dt);
            {
                std::lock_guard<std::mutex> lk(mtx_);
                pos_       = Eigen::Vector3d::Zero();
                pos_valid_ = false;
                radar_valid_ = false;
            }
            last_radar_t_ = rclcpp::Time(0, 0, RCL_ROS_TIME);  // fresh dt on next frame
            return;
        }

        // ── Position Update ───────────────────────────────────────────────
        updateWithDeadReckoning(v_base, dt);

        if (inliers_msg.width > 0)  radar_inlier_pub_->publish(inliers_msg);
        if (outliers_msg.width > 0) radar_outlier_pub_->publish(outliers_msg);
        if (dynamic_msg.width > 0)  radar_dynamic_pub_->publish(dynamic_msg);
    }


    // ── Dead-reckoning fallback (no factor graph) ─────────────────────────
    void updateWithDeadReckoning(const Eigen::Vector3d& v_base, double dt) {
        if (dt <= 0.0 || dt >= 2.0) return;
        std::lock_guard<std::mutex> lk(mtx_);

        // Update q_pose_ from calibrated IMU output (extrinsic applied)
        // so that position integration and published orientation are consistent.
        Eigen::Quaterniond q_cal;
        double ax_unused, ay_unused, az_unused, gx_unused, gy_unused, gz_unused;
        imu_proc_->getOutput(q_cal, ax_unused, ay_unused, az_unused,
                              gx_unused, gy_unused, gz_unused);
        q_pose_ = q_cal;

        // In 2D mode suppress vertical velocity before integrating — prevents
        // noisy vz Doppler estimates from accumulating into the Z position.
        Eigen::Vector3d v = v_base;
        if (use_2d_mode_) v.z() = 0.0;

        if (points_in_world_frame_) {
            // v is already in world frame — integrate directly
            pos_ += v * dt;
        } else {
            // v is in body frame — rotate to world frame
            pos_ += q_pose_.toRotationMatrix() * (v * dt);
        }
        pos_valid_ = true;
    }

    void publishCallback() {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!imu_proc_->isReady()) return;

        Eigen::Quaterniond q_imu;
        double ax, ay, az, gx, gy, gz;
        imu_proc_->getOutput(q_imu, ax, ay, az, gx, gy, gz);

        Eigen::Quaterniond q_raw;
        double qw, qx, qy, qz;
        imu_proc_->getRawQuaternion(qw, qx, qy, qz);
        q_raw = Eigen::Quaterniond(qw, qx, qy, qz);

        // Prefer the FG/DR accumulated Z once valid; use lidar only as initial
        // seed before any position update has occurred.  The raw lidar EMA
        // drifts at sensor-noise rate which is misleading in the published pose.
        // Negate Z for correct sign of vertical translation
        double z = pos_valid_ ? -pos_.z() :
                   (lidar_proc_->isValid() ? -lidar_proc_->getHeight() : 0.0);

        // When factor graph is active and has produced an optimised pose, use
        // q_pose_ (FG orientation) with the stored extrinsic applied so that
        // Use continuously-updated Madgwick IMU estimate.
        Eigen::Quaterniond q_for_odom = q_imu;

        const rclcpp::Time stamp = now();
        publishIMUPose(stamp, q_raw);
        publishFusedIMU(stamp, q_imu, gx, gy, gz, ax, ay, az);
        publishOdometry(stamp, pos_.x(), pos_.y(), z, q_for_odom);

        if (publish_tf_) {
            publishTF(stamp, q_for_odom, pos_.x(), pos_.y(), z);
        }
    }

    // ── Publishers ────────────────────────────────────────────────────────
    void publishIMUPose(const rclcpp::Time& stamp, const Eigen::Quaterniond& q) {
        geometry_msgs::msg::PoseWithCovarianceStamped msg;
        msg.header.stamp    = stamp;
        msg.header.frame_id = "imu_link";
        msg.pose.pose.orientation.w = q.w();
        msg.pose.pose.orientation.x = q.x();
        msg.pose.pose.orientation.y = q.y();
        msg.pose.pose.orientation.z = q.z();
        msg.pose.covariance[21] = 0.01;  // roll
        msg.pose.covariance[28] = 0.01;  // pitch
        msg.pose.covariance[35] = 1.0;   // yaw
        imu_pose_pub_->publish(msg);
    }

    void publishFusedIMU(const rclcpp::Time& stamp, const Eigen::Quaterniond& q,
                         double gx, double gy, double gz,
                         double ax, double ay, double az) {
        sensor_msgs::msg::Imu msg;
        msg.header.stamp    = stamp;
        msg.header.frame_id = base_frame_;
        msg.orientation.w   = q.w();
        msg.orientation.x   = q.x();
        msg.orientation.y   = q.y();
        msg.orientation.z   = q.z();
        msg.orientation_covariance         = {0.01, 0, 0,  0, 0.01, 0,  0, 0, 1.0};
        msg.angular_velocity.x             = gx;
        msg.angular_velocity.y             = gy;
        msg.angular_velocity.z             = gz;
        msg.angular_velocity_covariance    = {1e-4, 0, 0,  0, 1e-4, 0,  0, 0, 1e-4};
        msg.linear_acceleration.x          = ax;
        msg.linear_acceleration.y          = ay;
        msg.linear_acceleration.z          = az;
        msg.linear_acceleration_covariance = {0.01, 0, 0,  0, 0.01, 0,  0, 0, 0.01};
        fused_imu_pub_->publish(msg);
    }

    void publishOdometry(const rclcpp::Time& stamp,
                         double px, double py, double z,
                         const Eigen::Quaterniond& q) {
        nav_msgs::msg::Odometry odom;
        odom.header.stamp    = stamp;
        odom.header.frame_id = odom_frame_;
        odom.child_frame_id  = base_frame_;

        odom.pose.pose.position.x    = px;
        odom.pose.pose.position.y    = py;
        odom.pose.pose.position.z    = z;
        odom.pose.pose.orientation.w = q.w();
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();

        const double xy_cov = pos_valid_ ? 0.05 : 9999.0;
        const double  z_cov = lidar_proc_->isValid() ? 0.005 : 9999.0;
        odom.pose.covariance = {
            xy_cov, 0, 0, 0, 0, 0,
            0, xy_cov, 0, 0, 0, 0,
            0, 0, z_cov, 0, 0, 0,
            0, 0, 0, 0.01, 0, 0,
            0, 0, 0, 0, 0.01, 0,
            0, 0, 0, 0, 0, 1.0
        };

        if (radar_valid_) {
            odom.twist.twist.linear.x = radar_vel_.x();
            odom.twist.twist.linear.y = radar_vel_.y();
            odom.twist.twist.linear.z = radar_vel_.z();
            odom.twist.covariance[0]  = radar_sigma_.x() * radar_sigma_.x();
            odom.twist.covariance[7]  = radar_sigma_.y() * radar_sigma_.y();
            odom.twist.covariance[14] = radar_sigma_.z() * radar_sigma_.z();
            odom.twist.covariance[21] = 9999.0;
            odom.twist.covariance[28] = 9999.0;
            odom.twist.covariance[35] = 9999.0;
        } else {
            for (int i = 0; i < 6; ++i) odom.twist.covariance[i * 7] = 9999.0;
        }

        fused_odom_pub_->publish(odom);

        geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
        pose_msg.header = odom.header;
        pose_msg.pose   = odom.pose;
        fused_pose_pub_->publish(pose_msg);
    }

    void publishTF(const rclcpp::Time& stamp, const Eigen::Quaterniond& q,
                   double px, double py, double z) {
        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp    = stamp;
        tf.header.frame_id = odom_frame_;
        tf.child_frame_id  = base_frame_;
        tf.transform.translation.x = px;
        tf.transform.translation.y = py;
        tf.transform.translation.z = z;
        tf.transform.rotation.w    = q.w();
        tf.transform.rotation.x    = q.x();
        tf.transform.rotation.y    = q.y();
        tf.transform.rotation.z    = q.z();
        tf_broadcaster_->sendTransform(tf);
    }

    // ── Members ───────────────────────────────────────────────────────────
    std::mutex mtx_;

    // Gyro bias calibration (stationary warm-up before first motion)
    Eigen::Vector3d gyro_calib_sum_{Eigen::Vector3d::Zero()};
    int             gyro_calib_count_{0};
    int             gyro_calib_samples_{2400};  // ~5s at 480 Hz
    bool            gyro_bias_ready_{false};

    double publish_hz_{50.0};
    bool   publish_tf_{false};
    bool   use_2d_mode_{false};
    bool   is_holonomic_{false};
    std::string odom_frame_{"odom"}, base_frame_{"base_link"};
    std::string imu_topic_, lidar_topic_, radar_topic_;

    // Extrinsics
    Eigen::Matrix3d R_radar_to_base_{Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d R_imu_to_base_{Eigen::Matrix3d::Identity()};
    bool points_in_world_frame_{false};

    // Processors
    std::unique_ptr<IMUProcessor>     imu_proc_;
    std::unique_ptr<LidarProcessor>   lidar_proc_;
    std::unique_ptr<RadarProcessor>   radar_proc_;
    std::unique_ptr<RadarScanMatcher> scan_matcher_;

    rclcpp::Time last_imu_t_;

    // Radar state
    Eigen::Vector3d radar_vel_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d radar_sigma_{Eigen::Vector3d(9999, 9999, 9999)};
    bool            radar_valid_{false};
    double          max_radar_speed_{2.0};  // m/s — reject implausible RANSAC results

    // Accumulated pose (world frame) — updated via Doppler dead-reckoning
    Eigen::Vector3d    pos_{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond q_pose_{Eigen::Quaterniond::Identity()};
    bool               pos_valid_{false};

    // IMU bias and velocity state
    Eigen::Vector3d current_vel_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d current_b_acc_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d current_b_gyro_{Eigen::Vector3d::Zero()};

    // Per-frame radar timing
    rclcpp::Time last_radar_t_;

    // Bag loop reset
    bool   enable_bag_loop_reset_{false};
    double bag_loop_reset_threshold_sec_{1.0};

    // Subscriptions
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr        imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr       lidar_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr radar_sub_;

    // Publishers
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr imu_pose_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr                         fused_imu_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr                        fused_odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr fused_pose_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr                  radar_inlier_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr                  radar_outlier_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr                  radar_dynamic_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster>                               tf_broadcaster_;

    rclcpp::TimerBase::SharedPtr timer_;
};

// ============================================================================
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FusedOdomNode>());
    rclcpp::shutdown();
    return 0;
}
