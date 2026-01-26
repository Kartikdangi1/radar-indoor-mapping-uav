#!/usr/bin/env python3
"""
Radar Mapping Launch File v2
============================
With proper TF configuration for map accumulation.

Your TF tree (from view_frames):
  map -> odom -> base_link -> imu_link -> ARS_548
                          \-> radar_frame
                          \-> camera_link -> ...

The occupancy grid will be in 'odom' frame so it accumulates as robot moves.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    
    # Package directories
    radar_odom_dir = get_package_share_directory('radar_odom')
    temporal_mapping_dir = get_package_share_directory('temporal_radar_mapping')
    
    # Config files
    config_radar = os.path.join(radar_odom_dir, 'config', 'radar_pcl_processor.yaml')
    config_graph = os.path.join(radar_odom_dir, 'config', 'graph_slam.yaml')
    config_occupancy = os.path.join(temporal_mapping_dir, 'config', 'temporal_radar_occupancy_config.yaml')
    
    # Launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false')
    
    declare_launch_rviz = DeclareLaunchArgument(
        'launch_rviz', default_value='true')
    
    # ==========================================================================
    # STATIC TF PUBLISHERS
    # ==========================================================================
    # Your existing TF tree already has these, so we only add if missing.
    # Based on your view_frames output, these are already published by your robot.
    # If you get TF errors, uncomment the needed ones.
    
    # base_link -> imu_link (probably already from robot_state_publisher)
    # base_to_imu_tf = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='base_to_imu_tf',
    #     arguments=['--x', '0.0', '--y', '0.0', '--z', '0.0',
    #                '--roll', '0', '--pitch', '0', '--yaw', '0',
    #                '--frame-id', 'base_link', '--child-frame-id', 'imu_link'],
    # )
    
    # imu_link -> ARS_548 (probably already from your robot config)
    # imu_to_radar_tf = Node(...)
    
    # base_link -> radar_frame (for compatibility)
    base_to_radar_frame_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_radar_frame_tf',
        arguments=['--x', '0.0', '--y', '0.0', '--z', '0.1',
                   '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'base_link', '--child-frame-id', 'radar_frame'],
        output='screen'
    )
    
    # ==========================================================================
    # RADAR ODOMETRY PIPELINE
    # ==========================================================================
    
    radar_pcl_processor = Node(
        package='radar_odom',
        executable='radar_pcl_processor',
        name='radar_pcl_processor',
        output='screen',
        parameters=[config_radar],
    )
    
    optimizer = Node(
        package='radar_odom',
        executable='optimizer',
        name='graph_slam',
        output='screen',
        parameters=[config_graph],
    )
    
    # This publishes odom -> base_link from /odometry topic
    baselink_tf = Node(
        package='radar_odom',
        executable='baselink_tf',
        name='baselink_tf',
        output='screen',
        parameters=[{'topic_name': '/odometry'}]
    )
    
    record = Node(
        package='radar_odom',
        executable='record',
        name='record',
        output='screen'
    )
    
    # ==========================================================================
    # OCCUPANCY GRID MAPPING (v2 with TF transforms)
    # ==========================================================================
    
    occupancy_grid_node = Node(
        package='temporal_radar_mapping',
        executable='temporal_radar_occupancy_node',
        name='temporal_radar_occupancy_grid',
        output='screen',
        parameters=[
            config_occupancy,
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            # Override key parameters for your setup:
            {'map_frame': 'odom'},             # Grid accumulates in odom frame
            {'base_frame': 'base_link'},       # Robot base for ray casting
            {'radar_frame': 'ARS_548'},        # Your radar sensor frame
            {'input_topic': '/PointCloudDetection'},
            {'grid_topic': '/map'},
            {'grid_size_m': 50.0},
            {'grid_resolution': 0.25},
            # Persistence settings:
            {'frames_before_decay': 100},
            {'decay_factor': 0.995},
            {'min_observations_for_static': 3},
            {'occupied_protection_frames': 50},
            # Free space (conservative):
            {'mark_free_space': True},
            {'free_space_weight': 0.3},
            {'safety_margin_m': 0.5},
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )
    
    # ==========================================================================
    # RVIZ
    # ==========================================================================
    
    rviz_config_path = os.path.join(temporal_mapping_dir, 'rviz', 'radar_mapping.rviz')
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path] if os.path.exists(rviz_config_path) else [],
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        condition=IfCondition(LaunchConfiguration('launch_rviz'))
    )
    
    # ==========================================================================
    # LAUNCH
    # ==========================================================================
    
    return LaunchDescription([
        declare_use_sim_time,
        declare_launch_rviz,
        
        # TF (only add what's missing from your robot)
        base_to_radar_frame_tf,
        
        # Radar odometry
        radar_pcl_processor,
        optimizer,
        baselink_tf,
        record,
        
        # Mapping
        occupancy_grid_node,
        
        # Visualization
        # rviz_node,
    ])