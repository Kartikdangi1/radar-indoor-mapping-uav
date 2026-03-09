#!/usr/bin/env python3
"""
capture_pipeline_data.py
========================
Subscribes to live pipeline output topics during a bag playback run and
saves NPZ files on shutdown (SIGTERM or --duration timeout).

Captures topics that the pipeline computes at runtime (not in the raw bag):
  /fused_odom/odometry         → odom.npz
  /PointCloudDetectionFiltered → radar_odom.npz
  /PointCloudDetectionMapping  → radar_map.npz
  /fused_odom/loop_closures    → loop_closures.npz

Usage:
    python3 eval/capture_pipeline_data.py --out eval/data [--duration 300]
    # Kill with SIGTERM when bag finishes — data is saved on exit.
"""

import argparse
import os
import signal
import sys

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2
    from visualization_msgs.msg import MarkerArray
    from scipy.spatial.transform import Rotation
except ImportError:
    print("ERROR: rclpy not available. Source /opt/ros/humble/setup.bash")
    sys.exit(1)


class PipelineDataCapture(Node):
    def __init__(self, out_dir: str, duration: float):
        super().__init__("pipeline_data_capture")
        self.out_dir_ = out_dir
        self.duration_ = duration
        self.start_wall_ = __import__("time").time()
        self.data_ = {"odom": [], "radar_odom": [], "radar_map": [], "loop_closures": []}

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(Odometry,    "/fused_odom/odometry",         self._odom_cb,       qos)
        self.create_subscription(PointCloud2, "/PointCloudDetectionFiltered", self._radar_odom_cb, qos)
        self.create_subscription(PointCloud2, "/PointCloudDetectionMapping",  self._radar_map_cb,  qos)
        self.create_subscription(MarkerArray, "/fused_odom/loop_closures",    self._lc_cb,         qos)

        if duration > 0:
            self.create_timer(2.0, self._check_duration)

        self.get_logger().info(f"Capturing pipeline data → {out_dir}  (duration={duration:.0f}s)")

    def _stamp(self, header):
        return header.stamp.sec + header.stamp.nanosec * 1e-9

    def _odom_cb(self, msg):
        t = self._stamp(msg.header)
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        rpy = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_euler('xyz')
        self.data_["odom"].append([t, p.x, p.y, rpy[0], rpy[1], rpy[2], v.x, v.y])

    def _radar_odom_cb(self, msg):
        self.data_["radar_odom"].append([self._stamp(msg.header), float(msg.width * msg.height)])

    def _radar_map_cb(self, msg):
        self.data_["radar_map"].append([self._stamp(msg.header), float(msg.width * msg.height)])

    def _lc_cb(self, msg):
        if msg.markers:
            self.data_["loop_closures"].append([self._stamp(msg.markers[0].header)])

    def _check_duration(self):
        if __import__("time").time() - self.start_wall_ >= self.duration_:
            self.save()
            rclpy.shutdown()

    def save(self):
        os.makedirs(self.out_dir_, exist_ok=True)
        for key, rows in self.data_.items():
            if rows:
                arr = np.array(rows, dtype=np.float64)
                path = os.path.join(self.out_dir_, f"{key}.npz")
                np.savez_compressed(path, data=arr)
                print(f"  {key}: {arr.shape[0]} samples → {path}")
            else:
                print(f"  {key}: 0 samples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",      required=True,    help="Output directory for NPZ files")
    parser.add_argument("--duration", type=float, default=0,
                        help="Auto-shutdown after N wall-clock seconds (0 = run until killed)")
    args = parser.parse_args()

    rclpy.init()
    node = PipelineDataCapture(args.out, args.duration)

    def _on_signal(sig, frame):
        node.save()
        rclpy.try_shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT,  _on_signal)

    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        node.save()


if __name__ == "__main__":
    main()
