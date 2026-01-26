#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2_ros/transform_broadcaster.h"

class BaselinkTf : public rclcpp::Node {
public:
    BaselinkTf() : Node("baselink_tf") {
        
        this->declare_parameter<std::string>("topic_name", "/odometry");
        std::string topic_name = this->get_parameter("topic_name").as_string();

        // Dynamic Transform Broadcaster (not static!)
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        odom_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
            topic_name, 10, std::bind(&BaselinkTf::odomCallback, this, std::placeholders::_1)
        );

        RCLCPP_INFO(this->get_logger(), 
            "BaselinkTf initialized, publishing odom->base_link from topic: %s", 
            topic_name.c_str());
    }

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        // Publish odom → base_link transform from odometry message
        geometry_msgs::msg::TransformStamped transform_stamped;
        
        transform_stamped.header.stamp = msg->header.stamp;
        transform_stamped.header.frame_id = "odom";     // Parent: odom frame
        transform_stamped.child_frame_id = "base_link"; // Child: base_link
        
        // Copy pose from odometry message
        transform_stamped.transform.translation.x = msg->pose.pose.position.x;
        transform_stamped.transform.translation.y = msg->pose.pose.position.y;
        transform_stamped.transform.translation.z = msg->pose.pose.position.z;
        
        transform_stamped.transform.rotation = msg->pose.pose.orientation;

        // Publish the transform at the sensor rate
        tf_broadcaster_->sendTransform(transform_stamped);
    }

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BaselinkTf>());
    rclcpp::shutdown();
    return 0;
}