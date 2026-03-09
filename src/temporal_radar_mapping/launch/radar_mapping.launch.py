#!/usr/bin/env python3

"""
Fused Odometry Fused Odometry + Temporal Radar Mapping Launch File (v5)
===============================================================
Launches the full radar mapping pipeline with fused odometry integration:
  1. Fused Odometry radar preprocessing (odometry) + fused odometry (IMU + Radar + LiDAR)
  2. Separate radar preprocessing for mapping (different filter params)
  3. Temporal 2D Bayesian occupancy grid mapping with fused odom registration

Data flow:
  /PointCloudDetection (raw ARS_548)
      ↓ [radar_preprocessing_node]        → /PointCloudDetectionFiltered (for odom)
      ↓ [radar_preprocessing_mapping_node] → /PointCloudDetectionMapping (for mapping)
      ↓ [fused_odom_node]                → /fused_odom/odometry (50 Hz)
      ↓ [temporal_radar_occupancy_node]   → /radar/occupancy_grid
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ========================================================================
    # LAUNCH ARGUMENTS
    # ========================================================================

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time (set true when playing rosbags with --clock)'
    )

    declare_launch_rviz = DeclareLaunchArgument(
        'launch_rviz',
        default_value='false',
        description='Launch RViz2 for visualization'
    )

    declare_fused_odom_config = DeclareLaunchArgument(
        'fused_odom_config_file',
        default_value='fused_odom_config.yaml',
        description='Fused Odometry fusion config file name'
    )

    declare_preprocessing_config = DeclareLaunchArgument(
        'preprocessing_config_file',
        default_value='radar_preprocessing_node.yaml',
        description='Radar preprocessing config file name (for odometry)'
    )

    declare_mapping_preprocessing_config = DeclareLaunchArgument(
        'mapping_preprocessing_config_file',
        default_value='radar_preprocessing_mapping.yaml',
        description='Radar preprocessing config file name (for mapping)'
    )

    declare_mapping_config = DeclareLaunchArgument(
        'mapping_config_file',
        default_value='temporal_radar_occupancy_config.yaml',
        description='Temporal occupancy grid mapping config file name'
    )

    declare_slam_config = DeclareLaunchArgument(
        'slam_config_file',
        default_value='radar_scan_slam.yaml',
        description='Radar scan SLAM config file name'
    )

    declare_grid_topic = DeclareLaunchArgument(
        'grid_topic',
        default_value='/radar/occupancy_grid',
        description='Output occupancy grid topic'
    )

    declare_log_level = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level (debug, info, warn, error)'
    )

    declare_publish_tf = DeclareLaunchArgument(
        'publish_tf',
        default_value='true',
        description='Enable TF publishing in Fused Odometry node (odom → base_link)'
    )

    # ========================================================================
    # Fused Odometry: RADAR PREPROCESSING (for odometry)
    # ========================================================================

    fused_odom_pkg = FindPackageShare('fused_odometry')

    radar_preprocessing_node = Node(
        package='fused_odometry',
        executable='radar_preprocessing_node',
        name='radar_preprocessing_node',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                fused_odom_pkg, 'config',
                LaunchConfiguration('preprocessing_config_file')
            ]),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ]
    )

    # ========================================================================
    # V5: RADAR PREPROCESSING (for mapping — separate filter params)
    # ========================================================================
    # This is a second instance of the same preprocessing node, but with
    # different config (wider FoV, stricter RCS, tighter Z-slice).
    # Outputs /PointCloudDetectionMapping for the occupancy grid.

    radar_preprocessing_mapping_node = Node(
        package='fused_odometry',
        executable='radar_preprocessing_node',
        name='radar_preprocessing_mapping_node',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                fused_odom_pkg, 'config',
                LaunchConfiguration('mapping_preprocessing_config_file')
            ]),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ]
    )

    # ========================================================================
    # Fused Odometry: FUSED ODOMETRY (IMU + Radar Doppler + LiDAR height)
    # ========================================================================

    fused_odom_node = Node(
        package='fused_odometry',
        executable='fused_odom_node',
        name='fused_odom_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            PathJoinSubstitution([
                fused_odom_pkg, 'config',
                LaunchConfiguration('fused_odom_config_file')
            ]),
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'publish_tf': LaunchConfiguration('publish_tf')
            }
        ],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )

    # ========================================================================
    # RADAR SCAN SLAM (map→odom drift correction)
    # ========================================================================

    radar_scan_slam_node = Node(
        package='fused_odometry',
        executable='radar_scan_slam_node',
        name='radar_scan_slam_node',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                fused_odom_pkg, 'config',
                LaunchConfiguration('slam_config_file')
            ]),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
    )

    # ========================================================================
    # TEMPORAL RADAR 2D OCCUPANCY GRID MAPPING (v5 — fused odom)
    # ========================================================================

    mapping_pkg = FindPackageShare('temporal_radar_mapping')

    temporal_occupancy_node = Node(
        package='temporal_radar_mapping',
        executable='temporal_radar_occupancy_node',
        name='temporal_radar_occupancy_grid',
        output='screen',
        emulate_tty=True,
        parameters=[
            PathJoinSubstitution([
                mapping_pkg, 'config',
                LaunchConfiguration('mapping_config_file')
            ]),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        remappings=[
            ('/radar/occupancy_grid', LaunchConfiguration('grid_topic')),
        ],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )

    # ========================================================================
    # TF PUBLISHING
    # ========================================================================

    radar_static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='radar_frame_publisher',
        arguments=[
            '0.0', '0.0', '0.0',
            '0.0', '0.0', '3.14159',
            'base_link', 'radar_frame'
        ],
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    # ARS_548 frame — the raw radar publishes with frame_id "ARS_548"
    # Same transform as radar_frame (180° yaw for forward-facing ARS_548)
    ars548_static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='ars548_frame_publisher',
        arguments=[
            '0.0', '0.0', '0.0',
            '0.0', '0.0', '3.14159',
            'base_link', 'ARS_548'
        ],
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    # ========================================================================
    # RVIZ2 (Optional)
    # ========================================================================

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([
            mapping_pkg, 'rviz', 'radar_mapping.rviz'
        ])],
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        condition=IfCondition(LaunchConfiguration('launch_rviz'))
    )

    # ========================================================================
    # LAUNCH
    # ========================================================================

    return LaunchDescription([
        # Arguments
        declare_use_sim_time,
        declare_launch_rviz,
        declare_fused_odom_config,
        declare_preprocessing_config,
        declare_mapping_preprocessing_config,
        declare_mapping_config,
        declare_slam_config,
        declare_grid_topic,
        declare_log_level,
        declare_publish_tf,

        # Fused Odometry odometry pipeline
        radar_preprocessing_node,
        radar_preprocessing_mapping_node,   # V5: separate mapping filter
        fused_odom_node,

        # SLAM (map→odom drift correction)
        radar_scan_slam_node,

        # Mapping (v5: uses fused odom + mapping-filtered cloud)
        temporal_occupancy_node,

        # TF (static transforms only; dynamic TF from fused_odom_node)
        radar_static_tf,
        ars548_static_tf,

        # Visualization
        rviz_node,
    ])


# =============================================================================
# USAGE
# =============================================================================
#
# Basic (live sensor):
#   ros2 launch temporal_radar_mapping radar_mapping.launch.py
#
# With rosbag playback:
#   ros2 launch temporal_radar_mapping radar_mapping.launch.py use_sim_time:=true
#   ros2 bag play /path/to/bag --clock
#
# With RViz:
#   ros2 launch temporal_radar_mapping radar_mapping.launch.py launch_rviz:=true
#
# Debug logging:
#   ros2 launch temporal_radar_mapping radar_mapping.launch.py log_level:=debug
#
# =============================================================================
#
# EXPECTED TOPIC FLOW (v5)
# =============================================================================
#
#   /PointCloudDetection (raw ARS_548)
#       ↓ [radar_preprocessing_node]
#   /PointCloudDetectionFiltered (range/FoV/RCS filtered — for odometry)
#       ↓ [fused_odom_node]
#   /fused_odom/odometry (nav_msgs/Odometry @ 50 Hz)
#
#   /PointCloudDetection (raw ARS_548)
#       ↓ [radar_preprocessing_mapping_node]
#   /PointCloudDetectionMapping (indoor-tuned filtering — for mapping)
#       ↓ + /fused_odom/odometry
#       ↓ [temporal_radar_occupancy_node]
#   /radar/occupancy_grid (nav_msgs/OccupancyGrid)
#
# =============================================================================
