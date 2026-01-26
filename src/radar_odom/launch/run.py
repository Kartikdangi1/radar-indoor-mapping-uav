import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    package_dir = get_package_share_directory('radar_odom')
    config_radar = os.path.join(package_dir, 'config', 'radar_pcl_processor.yaml')
    config_graph = os.path.join(package_dir, 'config', 'graph_slam.yaml')
    
    # Static transform: imu_link to ARS_548 (radar frame)
    # Adjust x, y, z, roll, pitch, yaw based on your actual mounting
    # Arguments: x y z yaw pitch roll frame_id child_frame_id
    # Example: radar is 10cm above and aligned with IMU
    imu_to_radar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='imu_to_radar_tf',
        arguments=['0.0', '0.0', '0.1', '0', '0', '0', 'imu_link', 'ARS_548'],
        output='screen'
    )
    
    # Static transform: base_link to imu_link
    # Adjust based on your IMU mounting position
    base_to_imu_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_imu_tf',
        arguments=['0.0', '0.0', '0.0', '0', '0', '0', 'base_link', 'imu_link'],
        output='screen'
    )
    
    # Static transform: map to odom (identity for now)
    map_to_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_tf',
        arguments=['0.0', '0.0', '0.0', '0', '0', '0', 'map', 'odom'],
        output='screen'
    )

    radar_pcl_processor = Node(
        package='radar_odom',
        executable='radar_pcl_processor',
        output='screen',
        name='radar_pcl_processor',
        parameters=[config_radar]
    )

    optimizer = Node(
        package='radar_odom',
        executable='optimizer',
        output='screen',
        name='graph_slam',
        parameters=[config_graph]
    )

    baselink_tf = Node(
        package='radar_odom',
        executable='baselink_tf',
        name='baselink_tf',
        parameters=[{'topic_name': '/odometry'}]
    )

    record = Node(
        package='radar_odom',
        executable='record',
        name='record'
    )

    nodes_to_execute = [
        imu_to_radar_tf,
        base_to_imu_tf,
        map_to_odom_tf,
        radar_pcl_processor,
        optimizer,
        record,
        baselink_tf
    ]

    return LaunchDescription(nodes_to_execute)