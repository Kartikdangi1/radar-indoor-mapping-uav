cmake_minimum_required(VERSION 3.10)
project(temporal_radar_mapping)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Suppress Boost CMake policy warning
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

# ============================================================================
# DEPENDENCIES
# ============================================================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)
find_package(Eigen3 REQUIRED)

# ============================================================================
# TEMPORAL RADAR OCCUPANCY GRID NODE
# ============================================================================
add_executable(temporal_radar_occupancy_node
  src/temporal_radar_occupancy_grid.cpp
  src/grid_utilities.cpp
  src/odometry_handler.cpp
  src/sensor_model.cpp
  src/occlusion_filter.cpp
  src/radar_pipeline.cpp
  src/publishers.cpp
)

target_include_directories(temporal_radar_occupancy_node PRIVATE
  include
)

ament_target_dependencies(temporal_radar_occupancy_node
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
  pcl_conversions
  PCL
  Eigen3
)

install(TARGETS temporal_radar_occupancy_node
  DESTINATION lib/${PROJECT_NAME}
)

# ============================================================================
# INSTALL SCRIPTS
# ============================================================================
install(PROGRAMS
  scripts/map_saver.py
  DESTINATION lib/${PROJECT_NAME}
)

# ============================================================================
# INSTALL DIRECTORIES
# ============================================================================
install(DIRECTORY
  launch
  config
  rviz
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
