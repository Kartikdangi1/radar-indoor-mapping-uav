cmake_minimum_required(VERSION 3.8)
project(fused_odometry)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ars548_messages REQUIRED)

include_directories(include)

# ── Sensor Processor Library ──────────────────────────────────────────────
add_library(fused_odom_processors
  src/imu_processor.cpp
  src/lidar_processor.cpp
  src/radar_processor.cpp
  src/radar_scan_matcher.cpp
)
ament_target_dependencies(fused_odom_processors
  rclcpp sensor_msgs Eigen3 pcl_conversions pcl_ros
)
target_link_libraries(fused_odom_processors
  ${PCL_LIBRARIES}
)

# ── Simplified Modular Node ──────────────────────────────────────────────
add_executable(fused_odom_node src/fused_odom_node.cpp)
ament_target_dependencies(fused_odom_node
  rclcpp sensor_msgs nav_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs
  pcl_conversions pcl_ros Eigen3 ars548_messages
)
target_link_libraries(fused_odom_node fused_odom_processors ${PCL_LIBRARIES})

# ── Radar Preprocessing Node ─────────────────────────────────────────────
add_executable(radar_preprocessing_node src/radar_preprocessing_node.cpp)
ament_target_dependencies(radar_preprocessing_node
  rclcpp sensor_msgs pcl_conversions pcl_ros Eigen3
)
target_link_libraries(radar_preprocessing_node ${PCL_LIBRARIES})

# ── Radar Scan SLAM Node ───────────────────────────────────────────────
add_executable(radar_scan_slam_node src/radar_scan_slam_node.cpp)
ament_target_dependencies(radar_scan_slam_node
  rclcpp sensor_msgs nav_msgs geometry_msgs visualization_msgs tf2 tf2_ros tf2_geometry_msgs
  pcl_conversions pcl_ros Eigen3
)
target_link_libraries(radar_scan_slam_node ${PCL_LIBRARIES})

# ── Installation ────────────────────────────────────────────────────────
install(TARGETS
    fused_odom_node
    radar_preprocessing_node
    radar_scan_slam_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch)

ament_package()