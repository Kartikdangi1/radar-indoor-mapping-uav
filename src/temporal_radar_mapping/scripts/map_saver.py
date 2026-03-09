#!/usr/bin/env python3
"""
Map Saver Node for Radar Occupancy Grid
========================================
Subscribes to the occupancy grid topic and saves the map as PGM + YAML
(standard ROS map format compatible with nav2_map_server).

Save triggers:
  1. Service call:  ros2 service call /save_map std_srvs/srv/Trigger
  2. Keyboard:      Press Enter in the terminal to save
  3. Shutdown:      Automatically saves on Ctrl+C

Output files:  <output_dir>/<map_name>.pgm  +  <map_name>.yaml
"""

import os
import threading
import sys
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import Trigger


class MapSaver(Node):
    def __init__(self):
        super().__init__('map_saver')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('output_dir', os.path.expanduser('~/maps'))
        self.declare_parameter('map_name', 'radar_map')
        self.declare_parameter('save_on_shutdown', True)
        self.declare_parameter('threshold_occupied', 65)
        self.declare_parameter('threshold_free', 25)

        self.map_topic_ = self.get_parameter('map_topic').value
        self.output_dir_ = self.get_parameter('output_dir').value
        self.map_name_ = self.get_parameter('map_name').value
        self.save_on_shutdown_ = self.get_parameter('save_on_shutdown').value
        self.thresh_occ_ = self.get_parameter('threshold_occupied').value
        self.thresh_free_ = self.get_parameter('threshold_free').value

        self.latest_map_ = None
        self.map_lock_ = threading.Lock()
        self.save_count_ = 0

        # Subscription with volatile durability to match publisher
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        self.sub_ = self.create_subscription(
            OccupancyGrid, self.map_topic_, self.map_callback, qos)

        # Service trigger
        self.srv_ = self.create_service(Trigger, 'save_map', self.save_map_srv)

        os.makedirs(self.output_dir_, exist_ok=True)

        self.get_logger().info(f'Map Saver ready')
        self.get_logger().info(f'  Subscribed to: {self.map_topic_}')
        self.get_logger().info(f'  Output dir:    {self.output_dir_}')
        self.get_logger().info(f'  Save via:      ros2 service call /save_map std_srvs/srv/Trigger')
        self.get_logger().info(f'  Or press Enter in this terminal to save')

    def map_callback(self, msg: OccupancyGrid):
        with self.map_lock_:
            self.latest_map_ = msg

    def save_map_srv(self, request, response):
        ok, path = self.save_map()
        response.success = ok
        response.message = f'Saved to {path}' if ok else 'No map received yet'
        return response

    def save_map(self, suffix=''):
        with self.map_lock_:
            if self.latest_map_ is None:
                self.get_logger().warn('No map data received yet, cannot save')
                return False, ''
            grid = self.latest_map_

        width = grid.info.width
        height = grid.info.height
        resolution = grid.info.resolution
        origin = grid.info.origin

        # Build PGM image (row-major, origin at bottom-left in ROS)
        # OccupancyGrid values: -1 = unknown, 0-100 = probability
        # PGM values: 0 = black (occupied), 254 = white (free), 205 = grey (unknown)
        img = np.full((height, width), 205, dtype=np.uint8)

        data = np.array(grid.data, dtype=np.int8)
        data_2d = data.reshape((height, width))

        # Free cells
        img[data_2d >= 0] = 254  # default to free for known cells
        # Occupied cells
        img[(data_2d >= self.thresh_occ_) & (data_2d <= 100)] = 0
        # Free cells (explicitly)
        img[(data_2d >= 0) & (data_2d < self.thresh_free_)] = 254
        # In-between cells get a gradient
        between = (data_2d >= self.thresh_free_) & (data_2d < self.thresh_occ_)
        img[between] = (255 * (1.0 - data_2d[between] / 100.0)).astype(np.uint8)
        # Unknown cells
        img[data_2d < 0] = 205

        # Flip vertically (PGM origin is top-left, ROS is bottom-left)
        img = np.flipud(img)

        # Generate filename
        self.save_count_ += 1
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if suffix:
            base = f'{self.map_name_}_{suffix}_{timestamp}'
        else:
            base = f'{self.map_name_}_{timestamp}'

        pgm_path = os.path.join(self.output_dir_, f'{base}.pgm')
        yaml_path = os.path.join(self.output_dir_, f'{base}.yaml')

        # Write PGM (P5 binary format)
        with open(pgm_path, 'wb') as f:
            header = f'P5\n{width} {height}\n255\n'
            f.write(header.encode())
            f.write(img.tobytes())

        # Write YAML metadata
        ox = origin.position.x
        oy = origin.position.y
        # Extract yaw from quaternion
        q = origin.orientation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        yaml_content = (
            f'image: {base}.pgm\n'
            f'resolution: {resolution:.6f}\n'
            f'origin: [{ox:.6f}, {oy:.6f}, {yaw:.6f}]\n'
            f'negate: 0\n'
            f'occupied_thresh: 0.65\n'
            f'free_thresh: 0.25\n'
        )
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        self.get_logger().info(
            f'Map saved ({width}x{height}, {resolution:.3f}m/px): {pgm_path}')
        return True, pgm_path

    def shutdown_save(self):
        if self.save_on_shutdown_:
            self.get_logger().info('Shutdown: saving final map...')
            self.save_map(suffix='final')


def main(args=None):
    rclpy.init(args=args)
    node = MapSaver()

    # Keyboard listener thread — press Enter to save
    def keyboard_listener():
        try:
            while rclpy.ok():
                input()  # blocks until Enter
                node.get_logger().info('Enter pressed — saving map...')
                node.save_map()
        except EOFError:
            pass

    kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
    kb_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_save()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
