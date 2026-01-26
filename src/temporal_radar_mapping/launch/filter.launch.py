#!/usr/bin/env python3

"""
Indoor Radar Occupancy Grid Launch File
========================================
Launches enhanced indoor radar SLAM with all 11 features.
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
    
    declare_input_topic = DeclareLaunchArgument(
        'input_topic',
        default_value='/PointCloudDetection',
        description='Input radar PointCloud2 topic'
    )
    
    declare_grid_topic = DeclareLaunchArgument(
        'grid_topic',
        default_value='/radar/occupancy_grid',
        description='Output occupancy grid topic'
    )
    
    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value='temporal_radar_occupancy_config.yaml',
        description='Configuration file name (default, warehouse, office, factory, corridor)'
    )
    
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    declare_launch_rviz = DeclareLaunchArgument(
        'launch_rviz',
        default_value='false',
        description='Launch RViz for visualization'
    )

    # ========================================================================
    # INDOOR RADAR OCCUPANCY GRID NODE (Multi-threaded)
    # ========================================================================
    
    radar_slam_node = Node(
        package='temporal_radar_mapping',
        executable='temporal_radar_occupancy_node',
        name='temporal_radar_occupancy_grid',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('temporal_radar_mapping'),
                'config',
                LaunchConfiguration('config_file')
            ]),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        remappings=[
            ('/radar/pointcloud', LaunchConfiguration('input_topic')),
            ('/radar/occupancy_grid', LaunchConfiguration('grid_topic')),
        ],
        arguments=['--ros-args', '--log-level', 'info'],
        emulate_tty=True,
    )

    # ========================================================================
    # RADAR-IMU TF BROADCASTER (Dynamic Transform)
    # ========================================================================
    
    radar_imu_tf_broadcaster = Node(
        package='temporal_radar_mapping',
        executable='radar_tf_broadcaster_node',
        name='radar_imu_tf_broadcaster',
        output='screen',
        parameters=[
            {'radar_pose_topic': '/robot/global_pose'},
            {'imu_topic': '/robot/sensor/imu/data'},
            {'parent_frame': 'map'},
            {'child_frame': 'base_link'},
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )

    # ========================================================================
    # STATIC TF PUBLISHERS
    # ========================================================================
    
    # Radar sensor frame (if ARS_548 is not already published)
    radar_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='radar_frame_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'radar_frame'],
        output='screen'
    )

    # ========================================================================
    # RVIZ VISUALIZATION (Optional)
    # ========================================================================
    
    rviz_config = PathJoinSubstitution([
        FindPackageShare('temporal_radar_mapping'),
        'rviz',
        'indoor_radar_slam.rviz'
    ])
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        condition=IfCondition(LaunchConfiguration('launch_rviz'))
    )

    # ========================================================================
    # LAUNCH DESCRIPTION
    # ========================================================================
    
    return LaunchDescription([
        # Arguments
        declare_input_topic,
        declare_grid_topic,
        declare_config_file,
        declare_use_sim_time,
        declare_launch_rviz,
        
        # Nodes
        radar_slam_node,
        radar_imu_tf_broadcaster,  # NEW: Dynamic TF broadcaster
        radar_tf_node,
        rviz_node,
    ])
