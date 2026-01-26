#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include "Eigen/Dense"
#include <opencv2/core/core.hpp>
#include "rio_utils/radar_point_cloud.hpp"
#include "RadarEgoVel.hpp"
#include <chrono>
#include <vector>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Transform.h>

// Madgwick filter for orientation estimation from raw IMU data
class MadgwickFilter {
public:
    MadgwickFilter(float beta = 0.1f) : beta_(beta), q0_(1.0f), q1_(0.0f), q2_(0.0f), q3_(0.0f), initialized_(false) {}

    void update(float gx, float gy, float gz, float ax, float ay, float az, float dt) {
        float recipNorm;
        float s0, s1, s2, s3;
        float qDot1, qDot2, qDot3, qDot4;
        float _2q0, _2q1, _2q2, _2q3, _4q0, _4q1, _4q2, _8q1, _8q2, q0q0, q1q1, q2q2, q3q3;

        // Rate of change of quaternion from gyroscope
        qDot1 = 0.5f * (-q1_ * gx - q2_ * gy - q3_ * gz);
        qDot2 = 0.5f * (q0_ * gx + q2_ * gz - q3_ * gy);
        qDot3 = 0.5f * (q0_ * gy - q1_ * gz + q3_ * gx);
        qDot4 = 0.5f * (q0_ * gz + q1_ * gy - q2_ * gx);

        // Compute feedback only if accelerometer measurement valid
        if (!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f))) {
            // Normalise accelerometer measurement
            recipNorm = 1.0f / sqrtf(ax * ax + ay * ay + az * az);
            ax *= recipNorm;
            ay *= recipNorm;
            az *= recipNorm;

            // Auxiliary variables
            _2q0 = 2.0f * q0_;
            _2q1 = 2.0f * q1_;
            _2q2 = 2.0f * q2_;
            _2q3 = 2.0f * q3_;
            _4q0 = 4.0f * q0_;
            _4q1 = 4.0f * q1_;
            _4q2 = 4.0f * q2_;
            _8q1 = 8.0f * q1_;
            _8q2 = 8.0f * q2_;
            q0q0 = q0_ * q0_;
            q1q1 = q1_ * q1_;
            q2q2 = q2_ * q2_;
            q3q3 = q3_ * q3_;

            // Gradient descent corrective step
            s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
            s1 = _4q1 * q3q3 - _2q3 * ax + 4.0f * q0q0 * q1_ - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az;
            s2 = 4.0f * q0q0 * q2_ + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az;
            s3 = 4.0f * q1q1 * q3_ - _2q1 * ax + 4.0f * q2q2 * q3_ - _2q2 * ay;
            recipNorm = 1.0f / sqrtf(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3);
            s0 *= recipNorm;
            s1 *= recipNorm;
            s2 *= recipNorm;
            s3 *= recipNorm;

            // Apply feedback step
            qDot1 -= beta_ * s0;
            qDot2 -= beta_ * s1;
            qDot3 -= beta_ * s2;
            qDot4 -= beta_ * s3;
        }

        // Integrate rate of change of quaternion
        q0_ += qDot1 * dt;
        q1_ += qDot2 * dt;
        q2_ += qDot3 * dt;
        q3_ += qDot4 * dt;

        // Normalise quaternion
        recipNorm = 1.0f / sqrtf(q0_ * q0_ + q1_ * q1_ + q2_ * q2_ + q3_ * q3_);
        q0_ *= recipNorm;
        q1_ *= recipNorm;
        q2_ *= recipNorm;
        q3_ *= recipNorm;

        initialized_ = true;
    }

    void getQuaternion(double& w, double& x, double& y, double& z) const {
        w = q0_;
        x = q1_;
        y = q2_;
        z = q3_;
    }

    bool isInitialized() const { return initialized_; }

private:
    float beta_;
    float q0_, q1_, q2_, q3_;
    bool initialized_;
};

// RadarProcessor class handles point cloud processing, transformation between sensor frames,
// and ego-velocity estimation using radar data.
class RadarProcessor : public rclcpp::Node {
public:
    RadarProcessor() : Node("radar_pcl_processor"), last_imu_time_(-1.0) {
        // Retrieve parameters and set up communication channels.
        getParams();
        setupSubscribersAndPublishers();
        initializeTransformation();
    }

private:
    // Function to load parameters from the ROS2 parameter server.
    void getParams() {
        rio::RadarEgoVelocityEstimatorConfig config;

        // Retrieve ROS parameters with default values
        // Updated default topics to match your system
        this->declare_parameter<std::string>("imu_topic", "/robot/sensor/imu/data");
        this->declare_parameter<std::string>("radar_topic", "/PointCloudDetection");
        this->declare_parameter<bool>("enable_dynamic_object_removal", true);
        this->declare_parameter<bool>("holonomic_vehicle", true);
        this->declare_parameter<double>("distance_near_thresh", 0.1);
        this->declare_parameter<double>("distance_far_thresh", 80.0);
        this->declare_parameter<double>("z_low_thresh", -40.0);
        this->declare_parameter<double>("z_high_thresh", 100.0);
        
        // Channel indices for your radar format
        // Your radar: x(0), y(4), z(8), v(12), r(16), RCS(20), azimuth(21), elevation(25)
        this->declare_parameter<int>("intensity_channel", 5);  // RCS channel index
        this->declare_parameter<int>("doppler_channel", 4);    // v (velocity/doppler) channel index
        
        // Retrieve estimator configuration parameters with default values
        this->declare_parameter<double>("min_dist", 0.5);
        this->declare_parameter<double>("max_dist", 400.0);
        this->declare_parameter<double>("min_db", -20.0);  // Lowered for your RCS values
        this->declare_parameter<double>("elevation_thresh_deg", 50.0);
        this->declare_parameter<double>("azimuth_thresh_deg", 56.5);
        this->declare_parameter<double>("doppler_velocity_correction_factor", 1.0);
        this->declare_parameter<double>("thresh_zero_velocity", 0.05);
        this->declare_parameter<double>("allowed_outlier_percentage", 0.30);
        this->declare_parameter<double>("sigma_zero_velocity_x", 1.0e-03);
        this->declare_parameter<double>("sigma_zero_velocity_y", 3.2e-03);
        this->declare_parameter<double>("sigma_zero_velocity_z", 1.0e-02);
        this->declare_parameter<double>("sigma_offset_radar_x", 0.0);
        this->declare_parameter<double>("sigma_offset_radar_y", 0.0);
        this->declare_parameter<double>("sigma_offset_radar_z", 0.0);
        this->declare_parameter<double>("max_sigma_x", 0.2);
        this->declare_parameter<double>("max_sigma_y", 0.2);
        this->declare_parameter<double>("max_sigma_z", 0.2);
        this->declare_parameter<double>("max_r_cond", 0.2);
        this->declare_parameter<bool>("use_cholesky_instead_of_bdcsvd", false);
        this->declare_parameter<bool>("use_ransac", true);
        this->declare_parameter<double>("outlier_prob", 0.05);
        this->declare_parameter<double>("success_prob", 0.995);
        this->declare_parameter<double>("N_ransac_points", 5.0);
        this->declare_parameter<double>("inlier_thresh", 0.5);
        
        // Madgwick filter beta parameter (higher = more accelerometer influence)
        this->declare_parameter<double>("madgwick_beta", 0.1);
        
        this->get_parameter("imu_topic", imu_topic_);
        this->get_parameter("radar_topic", radar_topic_);
        this->get_parameter("enable_dynamic_object_removal", enable_dynamic_object_removal_);
        this->get_parameter("holonomic_vehicle", holonomic_vehicle_);
        this->get_parameter("distance_near_thresh", distance_near_thresh_);
        this->get_parameter("distance_far_thresh", distance_far_thresh_);
        this->get_parameter("z_low_thresh", z_low_thresh_);
        this->get_parameter("z_high_thresh", z_high_thresh_);
        this->get_parameter("intensity_channel", intensity_channel_);
        this->get_parameter("doppler_channel", doppler_channel_);

        // Retrieve estimator configuration parameters
        this->get_parameter("min_dist", config.min_dist);
        this->get_parameter("max_dist", config.max_dist);
        this->get_parameter("min_db", config.min_db);
        this->get_parameter("elevation_thresh_deg", config.elevation_thresh_deg);
        this->get_parameter("azimuth_thresh_deg", config.azimuth_thresh_deg);
        this->get_parameter("doppler_velocity_correction_factor", config.doppler_velocity_correction_factor);
        this->get_parameter("thresh_zero_velocity", config.thresh_zero_velocity);
        this->get_parameter("allowed_outlier_percentage", config.allowed_outlier_percentage);
        this->get_parameter("sigma_zero_velocity_x", config.sigma_zero_velocity_x);
        this->get_parameter("sigma_zero_velocity_y", config.sigma_zero_velocity_y);
        this->get_parameter("sigma_zero_velocity_z", config.sigma_zero_velocity_z);
        this->get_parameter("sigma_offset_radar_x", config.sigma_offset_radar_x);
        this->get_parameter("sigma_offset_radar_y", config.sigma_offset_radar_y);
        this->get_parameter("sigma_offset_radar_z", config.sigma_offset_radar_z);
        this->get_parameter("max_sigma_x", config.max_sigma_x);
        this->get_parameter("max_sigma_y", config.max_sigma_y);
        this->get_parameter("max_sigma_z", config.max_sigma_z);
        this->get_parameter("max_r_cond", config.max_r_cond);
        this->get_parameter("use_cholesky_instead_of_bdcsvd", config.use_cholesky_instead_of_bdcsvd);
        this->get_parameter("use_ransac", config.use_ransac);
        this->get_parameter("outlier_prob", config.outlier_prob);
        this->get_parameter("success_prob", config.success_prob);
        this->get_parameter("N_ransac_points", config.N_ransac_points);
        this->get_parameter("inlier_thresh", config.inlier_thresh);
        
        double madgwick_beta;
        this->get_parameter("madgwick_beta", madgwick_beta);
        madgwick_filter_ = std::make_shared<MadgwickFilter>(static_cast<float>(madgwick_beta));

        // Initialize the radar ego-velocity estimator with the retrieved configuration
        estimator_ = std::make_shared<rio::RadarEgoVel>(config);

        RCLCPP_INFO(this->get_logger(), "Holonomic: %s", holonomic_vehicle_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "IMU topic: %s", imu_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Radar topic: %s", radar_topic_.c_str());
    }

    // Set up ROS2 subscribers and publishers for IMU and radar point clouds.
    void setupSubscribersAndPublishers() {
        // Subscribe to IMU data
        imu_subscriber_ = create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, 10, [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
                imuCallback(msg);
            });

        // Subscribe to radar point cloud data
        radar_subscriber_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            radar_topic_, 10, [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                cloudCallback(msg);
            });

        // Publishers for processed data
        twist_publisher_ = create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>("/Ego_Vel_Twist", 5);
        radar_filtered_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_pointcloud", 10);
        inlier_pc2_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/inlier_pointcloud", 5);
        outlier_pc2_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/outlier_pointcloud", 5);
        raw_pc2_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/raw_pointcloud", 10);
        
        // Publisher for IMU with estimated orientation
        imu_with_orientation_publisher_ = create_publisher<sensor_msgs::msg::Imu>("/imu/data_with_orientation", 10);
    }

    // Initialize transformation matrices between different sensor frames
    void initializeTransformation() {
        // For your ARS_548 radar, we use a simpler transform
        // The radar frame is "ARS_548" and we transform to base_link/imu_link
        
        // Identity transform if radar and base_link are aligned
        // Modify this based on your actual mounting configuration
        // Assuming radar is mounted facing forward with minimal offset
        radar_to_base_link_ = (cv::Mat_<double>(4,4) <<
            1, 0, 0, 0,    // x forward
            0, 1, 0, 0,    // y left  
            0, 0, 1, 0.1,  // z up, 10cm above base_link (adjust as needed)
            0, 0, 0, 1);
    }

    // IMU callback - now computes orientation from gyro/accel using Madgwick filter
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr imu_msg) {
        double current_time = imu_msg->header.stamp.sec + imu_msg->header.stamp.nanosec * 1e-9;
        
        // Calculate dt
        double dt = 0.01;  // Default 100Hz
        if (last_imu_time_ > 0) {
            dt = current_time - last_imu_time_;
            if (dt <= 0 || dt > 1.0) {
                dt = 0.01;  // Reset to default if invalid
            }
        }
        last_imu_time_ = current_time;

        // Get gyro and accel data
        float gx = static_cast<float>(imu_msg->angular_velocity.x);
        float gy = static_cast<float>(imu_msg->angular_velocity.y);
        float gz = static_cast<float>(imu_msg->angular_velocity.z);
        float ax = static_cast<float>(imu_msg->linear_acceleration.x);
        float ay = static_cast<float>(imu_msg->linear_acceleration.y);
        float az = static_cast<float>(imu_msg->linear_acceleration.z);

        // Update Madgwick filter to estimate orientation
        madgwick_filter_->update(gx, gy, gz, ax, ay, az, static_cast<float>(dt));

        // Get estimated orientation
        double qw, qx, qy, qz;
        madgwick_filter_->getQuaternion(qw, qx, qy, qz);

        // Update current quaternion for use in cloud callback
        q_current_ = tf2::Quaternion(qx, qy, qz, qw);
        q_current_.normalize();
        
        // Publish IMU message with estimated orientation
        auto imu_with_orient = std::make_shared<sensor_msgs::msg::Imu>(*imu_msg);
        imu_with_orient->orientation.w = qw;
        imu_with_orient->orientation.x = qx;
        imu_with_orient->orientation.y = qy;
        imu_with_orient->orientation.z = qz;
        // Set covariance to indicate we have orientation data
        imu_with_orient->orientation_covariance[0] = 0.01;
        imu_with_orient->orientation_covariance[4] = 0.01;
        imu_with_orient->orientation_covariance[8] = 0.01;
        imu_with_orientation_publisher_->publish(*imu_with_orient);
    }

    // Radar point cloud callback for filtering and processing radar data
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr pcl_msg) {
        // Check if we have valid orientation from IMU
        if (!madgwick_filter_->isInitialized()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                "Waiting for IMU data to initialize orientation filter...");
            return;
        }

        // Parse PointCloud2 directly to preserve custom fields
        // Your radar format: x, y, z, v, r, RCS, azimuth, elevation
        RadarPointCloudType radar_point_raw;
        pcl::PointCloud<RadarPointCloudType>::Ptr radar_cloud_raw(new pcl::PointCloud<RadarPointCloudType>);

        // Find offsets for fields - adapted for your specific radar format
        int offset_x = -1, offset_y = -1, offset_z = -1;
        int offset_v = -1, offset_rcs = -1, offset_r = -1;
        int offset_azimuth = -1, offset_elevation = -1;
        uint8_t rcs_datatype = sensor_msgs::msg::PointField::FLOAT32;
        
        for (const auto &field : pcl_msg->fields) {
            if (field.name == "x") offset_x = field.offset;
            else if (field.name == "y") offset_y = field.offset;
            else if (field.name == "z") offset_z = field.offset;
            else if (field.name == "v") offset_v = field.offset;  // Doppler velocity
            else if (field.name == "r") offset_r = field.offset;  // Range
            else if (field.name == "RCS" || field.name == "rcs" || field.name == "intensity") {
                offset_rcs = field.offset;
                rcs_datatype = field.datatype;
            }
            else if (field.name == "azimuth") offset_azimuth = field.offset;
            else if (field.name == "elevation") offset_elevation = field.offset;
        }

        // Log field offsets for debugging (only once)
        static bool logged_fields = false;
        if (!logged_fields) {
            RCLCPP_INFO(this->get_logger(), "Point cloud fields found:");
            RCLCPP_INFO(this->get_logger(), "  x: %d, y: %d, z: %d", offset_x, offset_y, offset_z);
            RCLCPP_INFO(this->get_logger(), "  v: %d, r: %d, RCS: %d (type: %d)", 
                offset_v, offset_r, offset_rcs, rcs_datatype);
            RCLCPP_INFO(this->get_logger(), "  azimuth: %d, elevation: %d", offset_azimuth, offset_elevation);
            RCLCPP_INFO(this->get_logger(), "  point_step: %d", pcl_msg->point_step);
            logged_fields = true;
        }

        // Fallback to common float32 layout if explicit fields weren't present
        if (offset_x < 0) offset_x = 0;
        if (offset_y < 0) offset_y = 4;
        if (offset_z < 0) offset_z = 8;

        size_t point_count = static_cast<size_t>(pcl_msg->width) * static_cast<size_t>(pcl_msg->height);
        if (point_count == 0 && pcl_msg->point_step > 0) {
            point_count = pcl_msg->data.size() / pcl_msg->point_step;
        }

        for (size_t i = 0; i < point_count; ++i) {
            const uint8_t *data_ptr = &pcl_msg->data[i * pcl_msg->point_step];

            float x = *reinterpret_cast<const float *>(data_ptr + offset_x);
            float y = *reinterpret_cast<const float *>(data_ptr + offset_y);
            float z = *reinterpret_cast<const float *>(data_ptr + offset_z);

            if (std::isnan(x) || std::isnan(y) || std::isnan(z) ||
                std::isinf(x) || std::isinf(y) || std::isinf(z)) continue;

            float v = 0.0f;
            if (offset_v >= 0) {
                v = *reinterpret_cast<const float *>(data_ptr + offset_v);
            }

            float rcs_val = 0.0f;
            if (offset_rcs >= 0) {
                // Handle different RCS data types
                // Your radar has RCS as datatype 1 (UINT8)
                if (rcs_datatype == sensor_msgs::msg::PointField::FLOAT32) {
                    rcs_val = *reinterpret_cast<const float *>(data_ptr + offset_rcs);
                } else if (rcs_datatype == sensor_msgs::msg::PointField::UINT8) {
                    // UINT8 RCS - convert to dB scale if needed
                    rcs_val = static_cast<float>(*(data_ptr + offset_rcs));
                } else if (rcs_datatype == sensor_msgs::msg::PointField::INT8) {
                    rcs_val = static_cast<float>(*reinterpret_cast<const int8_t *>(data_ptr + offset_rcs));
                } else if (rcs_datatype == sensor_msgs::msg::PointField::INT32) {
                    rcs_val = static_cast<float>(*reinterpret_cast<const int32_t *>(data_ptr + offset_rcs));
                }
            }

            // Transform point from radar frame (ARS_548) to base_link frame
            cv::Mat pt_mat = (cv::Mat_<double>(4, 1) << x, y, z, 1);
            cv::Mat dst_mat = radar_to_base_link_ * pt_mat;

            radar_point_raw.x = dst_mat.at<double>(0, 0);
            radar_point_raw.y = dst_mat.at<double>(1, 0);
            radar_point_raw.z = dst_mat.at<double>(2, 0);
            radar_point_raw.intensity = rcs_val;
            radar_point_raw.doppler = v;

            radar_cloud_raw->points.push_back(radar_point_raw);
        }

        if (radar_cloud_raw->empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                "Received empty radar cloud after filtering");
            return;
        }

        // Publish the raw radar point cloud data
        sensor_msgs::msg::PointCloud2 pc2_raw_msg;
        pcl::toROSMsg(*radar_cloud_raw, pc2_raw_msg);
        pc2_raw_msg.header.stamp = pcl_msg->header.stamp;
        pc2_raw_msg.header.frame_id = "base_link";
        raw_pc2_publisher_->publish(pc2_raw_msg);

        // Estimate ego velocity based on the radar data
        Eigen::Vector3d v_radar, sigma_v_radar;
        sensor_msgs::msg::PointCloud2 inlier_radar_msg, outlier_radar_msg;

        if (initialization_) {
            q_previous_ = q_current_;
            initialization_ = false;
        }

        // Calculate rotation between previous and current orientations
        q_rotation_ = q_previous_.inverse() * q_current_;
        q_rotation_.normalize();
        tf2::Matrix3x3 q_rotation_matrix(q_rotation_);
        double roll, pitch, yaw;
        q_rotation_matrix.getRPY(roll, pitch, yaw);

        // Estimate ego velocity using the radar ego-velocity estimator
        if (estimator_->estimate(pc2_raw_msg, pitch, roll, yaw, holonomic_vehicle_, 
                                  v_radar, sigma_v_radar, inlier_radar_msg, outlier_radar_msg)) {
            // Publish the estimated twist with covariance
            geometry_msgs::msg::TwistWithCovarianceStamped twist_msg;
            twist_msg.header.stamp = pc2_raw_msg.header.stamp;
            twist_msg.header.frame_id = "base_link";
            twist_msg.twist.twist.linear.x = v_radar.x();
            twist_msg.twist.twist.linear.y = v_radar.y();
            twist_msg.twist.twist.linear.z = v_radar.z();
            
            // Set covariance from sigma values
            twist_msg.twist.covariance[0] = sigma_v_radar.x() * sigma_v_radar.x();  // xx
            twist_msg.twist.covariance[7] = sigma_v_radar.y() * sigma_v_radar.y();  // yy
            twist_msg.twist.covariance[14] = sigma_v_radar.z() * sigma_v_radar.z(); // zz
            
            twist_publisher_->publish(twist_msg);
            
            // Publish inlier/outlier clouds
            inlier_radar_msg.header.frame_id = "base_link";
            outlier_radar_msg.header.frame_id = "base_link";
            inlier_pc2_publisher_->publish(inlier_radar_msg);
            outlier_pc2_publisher_->publish(outlier_radar_msg);
        } else {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                "Velocity estimation failed. Points: %zu", radar_cloud_raw->size());
        }

        // Process inlier radar point cloud
        pcl::PointCloud<pcl::PointXYZI>::Ptr radar_cloud_inlier(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr radar_cloud_raw_(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(inlier_radar_msg, *radar_cloud_inlier);
        pcl::fromROSMsg(pc2_raw_msg, *radar_cloud_raw_);

        // Choose source cloud based on dynamic object removal flag
        pcl::PointCloud<pcl::PointXYZI>::ConstPtr source_cloud;
        if (enable_dynamic_object_removal_ && !radar_cloud_inlier->empty()) {
            source_cloud = radar_cloud_inlier;
        } else {
            source_cloud = radar_cloud_raw_;
        }

        if (source_cloud->empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                "Source cloud is empty. Skipping.");
            return;
        }

        // Apply distance and height filtering to the source cloud
        pcl::PointCloud<pcl::PointXYZI>::ConstPtr filtered_cloud = distanceFilter(source_cloud);
        sensor_msgs::msg::PointCloud2 filtered_cloud_msg;
        pcl::toROSMsg(*filtered_cloud, filtered_cloud_msg);
        filtered_cloud_msg.header.stamp = pc2_raw_msg.header.stamp;
        filtered_cloud_msg.header.frame_id = "base_link";
        radar_filtered_publisher_->publish(filtered_cloud_msg);

        // Update previous orientation
        q_previous_ = q_current_;
    }

    // Distance and height filter for point clouds
    pcl::PointCloud<pcl::PointXYZI>::ConstPtr distanceFilter(
        const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud) const {
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>);
        filtered->reserve(cloud->size());

        std::copy_if(cloud->begin(), cloud->end(), std::back_inserter(filtered->points), 
            [&](const pcl::PointXYZI& p) {
                double distance = p.getVector3fMap().norm();
                double z = p.z;
                return distance > distance_near_thresh_ && 
                       distance < distance_far_thresh_ && 
                       z < z_high_thresh_ && 
                       z > z_low_thresh_;
            });

        filtered->width = filtered->size();
        filtered->height = 1;
        filtered->is_dense = false;
        filtered->header = cloud->header;

        return filtered;
    }

    // Member variables
    std::string imu_topic_;
    std::string radar_topic_;
    int intensity_channel_;
    int doppler_channel_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr radar_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr radar_filtered_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr twist_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr inlier_pc2_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr outlier_pc2_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr raw_pc2_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_with_orientation_publisher_;

    cv::Mat radar_to_base_link_;
    std::shared_ptr<rio::RadarEgoVel> estimator_;
    std::shared_ptr<MadgwickFilter> madgwick_filter_;

    tf2::Quaternion q_previous_;
    tf2::Quaternion q_current_;
    tf2::Quaternion q_rotation_;

    double last_imu_time_;
    bool initialization_ = true;
    bool enable_dynamic_object_removal_;
    bool holonomic_vehicle_;
    double distance_near_thresh_;
    double distance_far_thresh_;
    double z_low_thresh_;
    double z_high_thresh_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RadarProcessor>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}