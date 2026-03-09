#!/usr/bin/env python3
"""
Quick pipeline sanity check for Radar Indoor Mapping UAV system.
Verifies all nodes are running, topics are publishing, and TF tree is valid.
Runtime: ~25 seconds.

Usage:
  # Terminal 1: launch the pipeline
  ros2 launch temporal_radar_mapping radar_mapping.launch.py use_sim_time:=true

  # Terminal 2: play the bag file
  ros2 bag play /path/to/bag --clock

  # Terminal 3: run this check
  python3 test_pipeline.py
"""

import sys
import time
import subprocess
from typing import List, Tuple
import rclpy
import rclpy.time
import rclpy.duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import MarkerArray
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class PipelineTester(Node):
    """ROS 2 node to collect messages and validate TF tree."""

    # Topics that must receive data for a PASS
    REQUIRED = ['fused_odometry', 'radar_filtered', 'radar_mapping', 'occupancy_map']
    # Topics that are informational only (OK if absent during short test)
    INFORMATIONAL = ['loop_closures', 'radar_inliers']

    def __init__(self):
        super().__init__('pipeline_tester')
        self.message_counts = {k: 0 for k in self.REQUIRED + self.INFORMATIONAL}

        # TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.create_subscription(Odometry, '/fused_odom/odometry',
                                 lambda m: self._count('fused_odometry'), sensor_qos)
        self.create_subscription(PointCloud2, '/PointCloudDetectionFiltered',
                                 lambda m: self._count('radar_filtered'), sensor_qos)
        self.create_subscription(PointCloud2, '/PointCloudDetectionMapping',
                                 lambda m: self._count('radar_mapping'), sensor_qos)
        self.create_subscription(OccupancyGrid, '/map',
                                 lambda m: self._count('occupancy_map'), reliable_qos)
        self.create_subscription(MarkerArray, '/fused_odom/loop_closures',
                                 lambda m: self._count('loop_closures'), reliable_qos)
        self.create_subscription(PointCloud2, '/fused_odom/radar_inliers',
                                 lambda m: self._count('radar_inliers'), sensor_qos)

    def _count(self, key: str):
        self.message_counts[key] += 1

    def check_tf_tree(self) -> Tuple[bool, dict]:
        """Check all required TF frames. Returns (all_ok, {frame_pair: ok})."""
        required = [('map', 'odom'), ('odom', 'base_link'), ('base_link', 'radar_frame')]
        results = {}
        for parent, child in required:
            try:
                self.tf_buffer.lookup_transform(
                    parent, child,
                    rclpy.time.Time(seconds=0),
                    timeout=rclpy.duration.Duration(seconds=0.2)
                )
                results[f'{parent}→{child}'] = True
            except Exception:
                results[f'{parent}→{child}'] = False
        return all(results.values()), results


def check_nodes(node_names: List[str]) -> Tuple[bool, dict]:
    """Check if required nodes are running via ros2 node list."""
    try:
        result = subprocess.run(['ros2', 'node', 'list'],
                                capture_output=True, text=True, timeout=5)
        running = result.stdout.strip().split('\n')
        running = [n.strip().lstrip('/') for n in running if n.strip()]
        results = {node: any(node in r for r in running) for node in node_names}
    except Exception as e:
        print(f"  Error querying nodes: {e}")
        results = {node: False for node in node_names}
    return all(results.values()), results


def main():
    print("=" * 65)
    print("  RADAR INDOOR MAPPING — QUICK PIPELINE CHECK")
    print("=" * 65)

    # [1] Node check (no ROS init needed)
    print("\n[1] Checking required nodes...")
    expected_nodes = [
        'fused_odom_node',
        'radar_preprocessing_node',
        'radar_scan_slam_node',
        'temporal_radar_occupancy',     # matches both _node and _grid variants
    ]
    nodes_ok, node_results = check_nodes(expected_nodes)
    for node, ok in node_results.items():
        print(f"  {'✓' if ok else '✗'} {node}")

    if not nodes_ok:
        print("\n  ✗ Pipeline not running. Launch with:")
        print("    ros2 launch temporal_radar_mapping radar_mapping.launch.py use_sim_time:=true")
        return False

    # [2] Collect messages
    rclpy.init()
    tester = PipelineTester()

    COLLECT_SEC = 20
    print(f"\n[2] Collecting messages for {COLLECT_SEC}s "
          f"(bag must be playing with --clock)...")

    start = time.time()
    last_print = -1
    while time.time() - start < COLLECT_SEC:
        rclpy.spin_once(tester, timeout_sec=0.05)
        elapsed = int(time.time() - start)
        if elapsed % 5 == 0 and elapsed != last_print:
            last_print = elapsed
            total = sum(tester.message_counts.values())
            print(f"  {elapsed:2d}s — total messages received: {total}")
    print(f"  {COLLECT_SEC}s — done\n")

    # [3] Message summary
    print("[3] Message reception:")
    messages_ok = True
    for key in tester.REQUIRED:
        n = tester.message_counts[key]
        ok = n > 0
        print(f"  {'✓' if ok else '✗'} {key:25s} {n:4d} msgs  {'[REQUIRED]' if not ok else ''}")
        if not ok:
            messages_ok = False
    for key in tester.INFORMATIONAL:
        n = tester.message_counts[key]
        print(f"  {'✓' if n > 0 else '–'} {key:25s} {n:4d} msgs  [info]")

    if not messages_ok:
        print("\n  ⚠ Missing required topics. Is the bag file playing?")
        print("    ros2 bag play /path/to/bag --clock")

    # [4] TF tree
    print("\n[4] TF tree validation:")
    tf_ok, tf_results = tester.check_tf_tree()
    for frame_pair, ok in tf_results.items():
        print(f"  {'✓' if ok else '✗'} {frame_pair}")
    if not tf_ok:
        print("  ⚠ Missing TF frames — IMU warmup may still be running (~10s)")

    # Summary
    all_ok = nodes_ok and messages_ok and tf_ok
    print("\n" + "=" * 65)
    if all_ok:
        print("  ✓ PIPELINE IS HEALTHY")
    else:
        print("  ✗ PIPELINE HAS ISSUES — see details above")
        if not nodes_ok:
            print("    → Launch the pipeline first")
        if not messages_ok:
            print("    → Start bag playback with --clock")
        if not tf_ok:
            print("    → Wait for IMU warmup to complete (~10s)")
    print("=" * 65)

    rclpy.shutdown()
    return all_ok


if __name__ == '__main__':
    try:
        success = main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        success = False
    sys.exit(0 if success else 1)
