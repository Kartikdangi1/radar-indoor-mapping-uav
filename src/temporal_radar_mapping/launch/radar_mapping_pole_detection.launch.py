#!/usr/bin/env python3

"""
Radar Mapping with Metal Pole Detection Optimization
=====================================================
Optimized variant of radar_mapping.launch.py for detecting thin vertical structures.

Key differences from standard mapping:
  1. Preprocessing RCS threshold lowered (-15 dB) to capture sparse pole returns
  2. Occupancy grid RCS gates relaxed (angle-dependent thresholds)
  3. Static confirmation lowered (5 vs 15 observations)
  4. Free-space clearing weight reduced (less aggressive clearing)
  5. Pole evidence protection increased

All other nodes/pipeline structure identical to standard launch.

USAGE:
  ros2 launch temporal_radar_mapping radar_mapping_pole_detection.launch.py use_sim_time:=true
  ros2 bag play /path/to/bag --clock
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ========================================================================
    # LAUNCH ARGUMENTS (same as standard radar_mapping.launch.py)
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
    # Fused Odometry: RADAR PREPROCESSING (for odometry — unchanged)
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
    # RADAR PREPROCESSING (for mapping — OPTIMIZED FOR POLE DETECTION)
    # ========================================================================
    # Lower RCS threshold to capture pole returns at off-perpendicular angles

    radar_preprocessing_mapping_node = Node(
        package='fused_odometry',
        executable='radar_preprocessing_node',
        name='radar_preprocessing_mapping_node',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                fused_odom_pkg, 'config', 'radar_preprocessing_mapping.yaml'
            ]),
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                # ===== POLE DETECTION OVERRIDE =====
                # Standard: -4.0 dB (only strong perpendicular reflections)
                # Poles: -15.0 dB (capture weak off-angle returns)
                'min_rcs_threshold': -15.0,
            }
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
    # RADAR SCAN SLAM (map→odom drift correction — unchanged)
    # ========================================================================

    radar_scan_slam_node = Node(
        package='fused_odometry',
        executable='radar_scan_slam_node',
        name='radar_scan_slam_node',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                fused_odom_pkg, 'config', 'radar_scan_slam.yaml'
            ]),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
    )

    # ========================================================================
    # TEMPORAL RADAR 2D OCCUPANCY GRID MAPPING (OPTIMIZED FOR POLES)
    # ========================================================================
    # Relaxed RCS gates, lower static threshold, reduced free-space clearing

    mapping_pkg = FindPackageShare('temporal_radar_mapping')

    temporal_occupancy_node = Node(
        package='temporal_radar_mapping',
        executable='temporal_radar_occupancy_node',
        name='temporal_radar_occupancy_grid',
        output='screen',
        emulate_tty=True,
        parameters=[
            PathJoinSubstitution([
                mapping_pkg, 'config', 'temporal_radar_occupancy_config.yaml'
            ]),
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                # ===== POLE DETECTION OVERRIDES =====
                # RCS thresholds (uint8): relax angle-dependent gates
                'min_rcs_u8_center': 20,        # was 35
                'min_rcs_u8_edge': 40,          # was 65
                'min_rcs_u8_high_edge': 65,     # was 95
                'min_rcs_u8_elevation': 50,     # was 75

                # Static confirmation: lower threshold for sparse pole hits
                'min_observations_for_static': 5,  # was 15

                # Free-space clearing: reduce weight to prevent erasing poles
                'free_space_weight': 0.20,      # was 0.40

                # Occupied cell protection: protect cells once they have evidence
                'occupied_protection_threshold': 0.40,  # was 0.20
                'occupied_protection_sec': 3.0,         # was 1.5

                # Confidence threshold: show poles with fewer observations
                'confidence_publish_threshold': 0.25,   # was 0.50

                # Log-odds ceiling: allow evidence to accumulate
                'lo_max': 3.5,                  # was 2.5
            }
        ],
        remappings=[
            ('/radar/occupancy_grid', '/radar/occupancy_grid'),
        ],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )

    # ========================================================================
    # TF PUBLISHING (unchanged)
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
        declare_log_level,
        declare_publish_tf,

        # Fused Odometry odometry pipeline
        radar_preprocessing_node,
        radar_preprocessing_mapping_node,   # POLE-OPTIMIZED: relaxed RCS threshold
        fused_odom_node,

        # SLAM
        radar_scan_slam_node,

        # Mapping (POLE-OPTIMIZED: relaxed RCS gates, static threshold, free weight)
        temporal_occupancy_node,

        # TF
        radar_static_tf,
        ars548_static_tf,

        # Visualization
        rviz_node,
    ])


# =============================================================================
# USAGE
# =============================================================================
#
# Basic (rosbag with simulation time):
#   ros2 launch temporal_radar_mapping radar_mapping_pole_detection.launch.py use_sim_time:=true
#   ros2 bag play /path/to/bag --clock
#
# With RViz:
#   ros2 launch temporal_radar_mapping radar_mapping_pole_detection.launch.py use_sim_time:=true launch_rviz:=true
#
# Debug logging:
#   ros2 launch temporal_radar_mapping radar_mapping_pole_detection.launch.py log_level:=debug use_sim_time:=true
#
# =============================================================================
#
# PARAMETER OVERRIDES FOR METAL POLE DETECTION
# =============================================================================
#
# Preprocessing (radar_preprocessing_mapping_node):
#   min_rcs_threshold: -15.0 (was -4.0)
#     → Capture pole returns at off-perpendicular angles
#
# Occupancy Grid (temporal_radar_occupancy_node):
#   min_rcs_u8_center: 20 (was 35)
#   min_rcs_u8_edge: 40 (was 65)
#   min_rcs_u8_high_edge: 65 (was 95)
#   min_rcs_u8_elevation: 50 (was 75)
#     → Relax angle-dependent RCS gates for sparse pole hits
#
#   min_observations_for_static: 5 (was 15)
#     → Static confirmation faster; once static, cells immune to free clearing
#
#   free_space_weight: 0.20 (was 0.40)
#     → Reduce aggressive free-space erasure of thin pole cells
#
#   occupied_protection_threshold: 0.40 (was 0.20)
#   occupied_protection_sec: 3.0 (was 1.5)
#     → Protect cells once they have meaningful evidence
#
#   confidence_publish_threshold: 0.25 (was 0.50)
#     → Show poles with fewer total observations
#
#   lo_max: 3.5 (was 2.5)
#     → Allow pole evidence to accumulate without early saturation
#
# =============================================================================
