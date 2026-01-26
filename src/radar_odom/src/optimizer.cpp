#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <Eigen/Dense>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/transform_broadcaster.h>
#include <ceres/ceres.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <ceres/CeresGraph.hpp>
#include <utils.hpp>
#include <queue>
#include <mutex>
#include <thread>
#include <deque>

class GraphSlam : public rclcpp::Node {
public:
    using PointT = pcl::PointXYZI;

    GraphSlam() : Node("graph_slam"), stop_processing_(false) {
        // Initialize parameters first
        InitializeParams();
        
        // Subscribers
        ego_vel_subscriber_ = this->create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
            "/Ego_Vel_Twist", 3000,
            [this](const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                twist_queue_.push(msg);
            });

        imu_subscriber_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu/data_with_orientation", 1024,
            [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                imu_queue_.push(msg);
            });

        point_cloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/filtered_pointcloud", 3000,
            [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                point_cloud_queue_.push(msg);
            });

        // Publishers
        odometry_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/odometry", 10);
        
        // NEW: Continuous odometry publisher (raw, unoptimized but smooth)
        continuous_odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/odometry_continuous", 50);
        
        // NEW: Path publisher for visualization
        path_publisher_ = this->create_publisher<nav_msgs::msg::Path>("/trajectory", 10);
        
        point_cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/keyframe_cloud", 10);

        // NEW: TF broadcaster for map→odom transform (loop-closure corrected)
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        // Initialize path message
        path_msg_.header.frame_id = "map";

        // Start processing thread
        processing_thread_ = std::thread(&GraphSlam::processMessages, this);
        
        RCLCPP_INFO(this->get_logger(), "GraphSlam node initialized");
        RCLCPP_INFO(this->get_logger(), "  Keyframe delta trans: %.2f m", keyframe_delta_trans_);
        RCLCPP_INFO(this->get_logger(), "  Keyframe delta angle: %.2f rad", keyframe_delta_angle_);
        RCLCPP_INFO(this->get_logger(), "  Max window size: %d", max_window_size_);
        RCLCPP_INFO(this->get_logger(), "  Min points for keyframe: %zu", min_points_for_keyframe_);
        RCLCPP_INFO(this->get_logger(), "  Max ICP fitness: %.1f", max_icp_fitness_);
        RCLCPP_INFO(this->get_logger(), "  Publishing continuous odometry on /odometry_continuous");
        RCLCPP_INFO(this->get_logger(), "  Publishing trajectory on /trajectory");
    }

    ~GraphSlam() {
        stop_processing_ = true;

        if (!Keyframes.empty()) {
            publishRemainingKeyframes();
        }
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
    }

private:
    // Main processing loop
    void processMessages() {
        while (!stop_processing_) {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (!twist_queue_.empty() && !imu_queue_.empty() && !point_cloud_queue_.empty()) {
                auto twist_msg = twist_queue_.front();
                auto imu_msg = imu_queue_.front();
                auto cloud_msg = point_cloud_queue_.front();
                lock.unlock();

                if (isTwistBeforeImu(twist_msg, imu_msg) && isImuBeforePointCloud(imu_msg, cloud_msg)) {
                    {
                        std::lock_guard<std::mutex> lock(queue_mutex_);
                        twist_queue_.pop();
                        imu_queue_.pop();
                        point_cloud_queue_.pop();
                    }
                    processOdometry(twist_msg, imu_msg, cloud_msg);
                } else {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    if (!isTwistBeforeImu(twist_msg, imu_msg)) {
                        imu_queue_.pop();
                    } else if (!isImuBeforePointCloud(imu_msg, cloud_msg)) {
                        point_cloud_queue_.pop();
                    }
                }
            } else {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    }

    // Check message timestamps for synchronization
    bool isTwistBeforeImu(
        const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& twist_msg,
        const sensor_msgs::msg::Imu::SharedPtr& imu_msg) {
        auto twist_time = twist_msg->header.stamp;
        auto imu_time = imu_msg->header.stamp;
        return (twist_time.sec < imu_time.sec) ||
               (twist_time.sec == imu_time.sec && twist_time.nanosec < imu_time.nanosec);
    }

    bool isImuBeforePointCloud(
        const sensor_msgs::msg::Imu::SharedPtr& imu_msg,
        const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg) {
        auto imu_time = imu_msg->header.stamp;
        auto cloud_time = cloud_msg->header.stamp;
        return (imu_time.sec < cloud_time.sec) ||
               (imu_time.sec == cloud_time.sec && imu_time.nanosec < cloud_time.nanosec);
    }

    // NEW: Publish map→odom transform from graph optimization
    void publishMapToOdomTransform(const builtin_interfaces::msg::Time& stamp) {
        if (Keyframes.empty()) {
            return;
        }

        // Get the optimized pose of the first keyframe in global (map) frame
        Eigen::Matrix4d first_kf_pose = Keyframes[0].getPose_matrix4d();
        tf2::Transform first_kf_transform = matrixToTransform(first_kf_pose);

        // The map→odom transform is the first keyframe's pose in map frame
        // This represents the global correction/offset from loop closures
        geometry_msgs::msg::TransformStamped map_to_odom_tf;
        map_to_odom_tf.header.stamp = stamp;
        map_to_odom_tf.header.frame_id = "map";
        map_to_odom_tf.child_frame_id = "odom";

        map_to_odom_tf.transform.translation.x = first_kf_transform.getOrigin().getX();
        map_to_odom_tf.transform.translation.y = first_kf_transform.getOrigin().getY();
        map_to_odom_tf.transform.translation.z = first_kf_transform.getOrigin().getZ();
        map_to_odom_tf.transform.rotation = tf2::toMsg(first_kf_transform.getRotation().normalize());

        tf_broadcaster_->sendTransform(map_to_odom_tf);
    }

    // NEW: Publish continuous odometry (called every frame, not just keyframes)
    void publishContinuousOdometry(const builtin_interfaces::msg::Time& stamp) {
        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header.frame_id = "map";
        odom_msg.child_frame_id = "base_link";
        odom_msg.header.stamp = stamp;

        // Use the current estimated pose (possibly corrected by optimization)
        tf2::Transform current_pose = getCorrectedPose();

        odom_msg.pose.pose.position.x = current_pose.getOrigin().getX();
        odom_msg.pose.pose.position.y = current_pose.getOrigin().getY();
        odom_msg.pose.pose.position.z = current_pose.getOrigin().getZ();
        odom_msg.pose.pose.orientation = tf2::toMsg(current_pose.getRotation().normalize());

        // Set covariance (approximate)
        odom_msg.pose.covariance[0] = 0.1;   // x
        odom_msg.pose.covariance[7] = 0.1;   // y
        odom_msg.pose.covariance[14] = 0.1;  // z
        odom_msg.pose.covariance[35] = 0.05; // yaw

        continuous_odom_publisher_->publish(odom_msg);

        // Also add to path for visualization
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header = odom_msg.header;
        pose_stamped.pose = odom_msg.pose.pose;
        
        // Limit path size to avoid memory issues
        if (path_msg_.poses.size() > 5000) {
            path_msg_.poses.erase(path_msg_.poses.begin(), 
                                   path_msg_.poses.begin() + 1000);
        }
        path_msg_.poses.push_back(pose_stamped);
        path_msg_.header.stamp = stamp;
        path_publisher_->publish(path_msg_);
    }

    // NEW: Get the corrected pose based on last keyframe optimization
    tf2::Transform getCorrectedPose() {
        if (Keyframes.empty()) {
            return estimated_pose_;
        }

        // Get the last optimized keyframe pose
        tf2::Transform last_kf_optimized = matrixToTransform(Keyframes.back().getPose_matrix4d());
        
        // Calculate the delta from last keyframe pose to current estimated pose
        tf2::Transform delta = keyframe_pose_.inverse() * estimated_pose_;
        
        // Apply delta to the optimized keyframe pose
        return last_kf_optimized * delta;
    }

    // Process odometry 
    void processOdometry(
        const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& twist_msg,
        const sensor_msgs::msg::Imu::SharedPtr& imu_msg,
        const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg) {

        pcl::PointCloud<PointT>::Ptr actual_cloud_(new pcl::PointCloud<PointT>);
        pcl::fromROSMsg(*cloud_msg, *actual_cloud_);

        // Check if orientation is valid
        double qw = imu_msg->orientation.w;
        double qx = imu_msg->orientation.x;
        double qy = imu_msg->orientation.y;
        double qz = imu_msg->orientation.z;
        
        double qnorm = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
        if (qnorm < 0.1) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Invalid IMU orientation (norm=%.3f). Waiting for valid data...", qnorm);
            return;
        }

        tf2::fromMsg(imu_msg->orientation, q_imu_);
        q_imu_.normalize();

        if (initialization_) {
            initialization_ = false;
            q_prev_ = q_imu_;
            keyframe_pose_ = estimated_pose_;  // Initialize keyframe pose
            RCLCPP_INFO(this->get_logger(), "Odometry initialized with first valid IMU orientation");
        }

        double this_time = imu_msg->header.stamp.sec + imu_msg->header.stamp.nanosec * 1e-9;
        static double last_time = this_time;
        double dt_imu = this_time - last_time;
        
        if (dt_imu <= 0 || dt_imu > 1.0) {
            dt_imu = 0.01;
        }
        last_time = this_time;

        double yaw_vel = imu_msg->angular_velocity.z * dt_imu;
        yaw_integrated_ += yaw_vel;
        yaw_integrated_kf_ += yaw_integrated_;

        update_estimated_pose(twist_msg, dt_imu);

        // NEW: Always publish continuous odometry for smooth visualization
        publishContinuousOdometry(imu_msg->header.stamp);

        if (decide(estimated_pose_)) {
            size_t cloud_size = actual_cloud_->size();
            
            if (cloud_size < min_points_for_keyframe_) {
                RCLCPP_WARN(this->get_logger(), 
                    "Skipping keyframe: cloud too small (%zu points, need %zu)", 
                    cloud_size, min_points_for_keyframe_);
            } else {
                keyframe_pose_ = estimated_pose_;
                keyframe_index_++;
                
                KeyFrame actual_keyframe(keyframe_index_, transformToMatrix(keyframe_pose_), 
                                         actual_cloud_, cloud_msg->header.stamp);
                Keyframes.push_back(actual_keyframe);
                
                yaw_integrated_kf_ = 0.0;
                RCLCPP_INFO(this->get_logger(), "Keyframe %zu created with %zu points", 
                           keyframe_index_, cloud_size);

                if (Keyframes.size() > 1) {
                    Eigen::Matrix4d prev_pose = Keyframes[Keyframes.size() - 2].getPose_matrix4d();
                    Eigen::Matrix4d actual_pose = Keyframes.back().getPose_matrix4d();
                    Eigen::Matrix4d odom_tf = prev_pose.inverse() * actual_pose;
                    Keyframes.back().add_Odom_tf(odom_tf);

                    CeresGraph ceres_graph(&Keyframes, max_window_size_);
                    
                    // Set ICP parameters
                    ceres_graph.setICPParams(min_points_for_icp_, max_icp_fitness_, 
                                            icp_max_correspondence_dist_);

                    int window_size = std::min(static_cast<int>(Keyframes.size()), max_window_size_);
                    ceres_graph.update_constraints(window_size);
                    ceres_graph.create_graph(window_size);
                    ceres_graph.optimize_graph(window_size);

                    // NEW: Publish map→odom transform (loop-closure corrected) from first keyframe
                    // This is the global correction from SLAM loop closures
                    if (Keyframes.size() > 0) {
                        publishMapToOdomTransform(imu_msg->header.stamp);
                    }

                    // Publish optimized keyframes (for reference/debugging)
                    size_t start_idx = Keyframes.size() > static_cast<size_t>(max_window_size_) 
                        ? Keyframes.size() - max_window_size_ : 0;

                    for (size_t i = start_idx; i < Keyframes.size(); ++i) {
                        KeyFrame& published_keyframe = Keyframes[i];

                        nav_msgs::msg::Odometry odometry_msg;
                        odometry_msg.header.frame_id = "map";
                        odometry_msg.child_frame_id = "base_link";
                        odometry_msg.header.stamp = published_keyframe.getTimestamp();

                        tf2::Transform odom_opt = matrixToTransform(published_keyframe.getPose_matrix4d());
                        odometry_msg.pose.pose.position.x = odom_opt.getOrigin().getX();
                        odometry_msg.pose.pose.position.y = odom_opt.getOrigin().getY();
                        odometry_msg.pose.pose.position.z = odom_opt.getOrigin().getZ();
                        odometry_msg.pose.pose.orientation = tf2::toMsg(odom_opt.getRotation().normalize());

                        odometry_publisher_->publish(odometry_msg);

                        sensor_msgs::msg::PointCloud2 keyframe_cloud_msg;
                        pcl::toROSMsg(*published_keyframe.getPointCloud(), keyframe_cloud_msg);
                        keyframe_cloud_msg.header.frame_id = "base_link";
                        keyframe_cloud_msg.header.stamp = imu_msg->header.stamp;

                        point_cloud_publisher_->publish(keyframe_cloud_msg);
                    }
                }
            }
        }

        yaw_integrated_ = 0.0;
        q_prev_ = q_imu_;
    }

    // Update the estimated pose using velocity and IMU data
    void update_estimated_pose(
        const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& twist_msg, 
        double dt) {
       
        double translation_x = twist_msg->twist.twist.linear.x * dt;
        double translation_y = twist_msg->twist.twist.linear.y * dt;
        double translation_z = twist_msg->twist.twist.linear.z * dt;
        tf2::Vector3 translation(translation_x, translation_y, translation_z);

        q_rot_ = q_prev_.inverse() * q_imu_;
        q_rot_.normalize();
        q_prev_ = q_imu_;

        tf2::Matrix3x3 m(q_rot_);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        
        tf2::Quaternion q_yaw;
        q_yaw.setRPY(roll, pitch, yaw_integrated_);
        q_yaw.normalize();

        tf2::Transform transform;
        transform.setRotation(q_yaw);
        transform.setOrigin(translation);

        estimated_pose_ = estimated_pose_ * transform;
        
        tf2::Quaternion q_current;
        tf2::Matrix3x3(estimated_pose_.getRotation()).getRotation(q_current);
        q_current.normalize();

        double roll_current, pitch_current, yaw_current;
        tf2::Matrix3x3(q_current).getRPY(roll_current, pitch_current, yaw_current);
        
        double roll_imu, pitch_imu, yaw_imu;
        tf2::Matrix3x3(q_imu_).getRPY(roll_imu, pitch_imu, yaw_imu);

        tf2::Quaternion q_combined;
        q_combined.setRPY(roll_imu - bias_.x(), pitch_imu - bias_.y(), yaw_current);
        q_combined.normalize();
   
        estimated_pose_.setRotation(q_combined);

        if (!validatePose(estimated_pose_)) {
            RCLCPP_ERROR(this->get_logger(), "Error: estimated pose contains NaN values");
        }
    }

    bool validatePose(const tf2::Transform& pose) {
        return !(std::isnan(pose.getOrigin().getX()) ||
                 std::isnan(pose.getOrigin().getY()) ||
                 std::isnan(pose.getOrigin().getZ()) ||
                 std::isnan(pose.getRotation().x()) ||
                 std::isnan(pose.getRotation().y()) ||
                 std::isnan(pose.getRotation().z()) ||
                 std::isnan(pose.getRotation().w()));
    }

    void InitializeParams() {
        // Keyframe parameters - REDUCED for sparse radar
        keyframe_delta_trans_ = this->declare_parameter<double>("keyframe_delta_trans", 0.3);
        keyframe_delta_angle_ = this->declare_parameter<double>("keyframe_delta_angle", 0.15);
        max_window_size_ = this->declare_parameter<int>("max_window_size", 5);
        
        // ICP parameters - now configurable at runtime!
        min_points_for_keyframe_ = this->declare_parameter<int>("min_points_for_keyframe", 10);
        min_points_for_icp_ = this->declare_parameter<int>("min_points_for_icp", 8);
        max_icp_fitness_ = this->declare_parameter<double>("max_icp_fitness", 10.0);
        icp_max_correspondence_dist_ = this->declare_parameter<double>("icp_max_correspondence_dist", 3.0);

        std::vector<double> bias_values;
        bias_values = this->declare_parameter<std::vector<double>>("bias_rpy", {0.00001, 0.0, 0.0});
        
        RCLCPP_INFO(this->get_logger(), "Bias RPY: [%.6f, %.6f, %.6f]", 
                    bias_values[0], bias_values[1], bias_values[2]);

        bias_(0) = bias_values[0];
        bias_(1) = bias_values[1];
        bias_(2) = bias_values[2];

        initialization_ = true;
        is_first_ = true;
        estimated_pose_ = tf2::Transform::getIdentity();
        keyframe_pose_ = tf2::Transform::getIdentity();
        yaw_integrated_ = 0.0;
        yaw_integrated_kf_ = 0.0;
        keyframe_index_ = 0;
    }

    bool decide(const tf2::Transform& Pose) {
        if (is_first_) {
            is_first_ = false;
            return true;
        }

        tf2::Transform delta = keyframe_pose_.inverse() * Pose;
        tf2::Vector3 translation = delta.getOrigin();
        tf2::Quaternion rotation = delta.getRotation();
        double dx = translation.length();

        double roll, pitch, yaw;
        tf2::Matrix3x3(rotation).getRPY(roll, pitch, yaw);

        return (dx >= keyframe_delta_trans_ || std::abs(yaw) >= keyframe_delta_angle_);
    }

    void publishRemainingKeyframes() {
        size_t start_idx = std::max(0, static_cast<int>(Keyframes.size()) - 
                                       static_cast<int>(max_window_size_));

        for (size_t i = start_idx; i < Keyframes.size(); ++i) {
            nav_msgs::msg::Odometry odometry_msg;
            odometry_msg.header.frame_id = "map";
            odometry_msg.child_frame_id = "base_link";
            odometry_msg.header.stamp = Keyframes[i].getTimestamp();

            tf2::Transform odom_opt = matrixToTransform(Keyframes[i].getPose_matrix4d());
            odometry_msg.pose.pose.position.x = odom_opt.getOrigin().getX();
            odometry_msg.pose.pose.position.y = odom_opt.getOrigin().getY();
            odometry_msg.pose.pose.position.z = odom_opt.getOrigin().getZ();
            odometry_msg.pose.pose.orientation = tf2::toMsg(odom_opt.getRotation().normalize());

            odometry_publisher_->publish(odometry_msg);
        }
    }

    // Member variables
    rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr ego_vel_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_subscriber_;
    
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr continuous_odom_publisher_;  // NEW
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;  // NEW
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_publisher_;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;  // NEW: For map→odom TF

    std::queue<geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr> twist_queue_;
    std::queue<sensor_msgs::msg::Imu::SharedPtr> imu_queue_;
    std::queue<sensor_msgs::msg::PointCloud2::SharedPtr> point_cloud_queue_;
    std::mutex queue_mutex_;

    tf2::Transform estimated_pose_;
    tf2::Transform keyframe_pose_;
    double yaw_integrated_;
    double yaw_integrated_kf_;

    tf2::Quaternion q_imu_;
    tf2::Quaternion q_prev_;
    tf2::Quaternion q_rot_;
    std::vector<KeyFrame> Keyframes;
    Eigen::Vector3d bias_;

    nav_msgs::msg::Path path_msg_;  // NEW: Store path for visualization

    // Parameters
    bool initialization_;
    bool is_first_;
    double keyframe_delta_trans_;
    double keyframe_delta_angle_;
    size_t keyframe_index_;
    int max_window_size_;
    size_t min_points_for_keyframe_;
    size_t min_points_for_icp_;
    double max_icp_fitness_;
    double icp_max_correspondence_dist_;

    std::thread processing_thread_;
    std::atomic<bool> stop_processing_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GraphSlam>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}