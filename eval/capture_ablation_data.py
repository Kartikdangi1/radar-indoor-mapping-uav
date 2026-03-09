#!/usr/bin/env python3
"""
capture_ablation_data.py
========================
Short-lived ROS 2 node that captures the *last* received /map and all
/fused_odom/odometry messages during an ablation run and saves them to disk.

Outputs (in --out directory):
    map.pgm          — final occupancy grid as PGM (format compatible with RViz)
    map_meta.yaml    — ROS OccupancyGrid metadata (resolution, origin, size)
    odom.npz         — odometry array (t, x, y, roll, pitch, yaw, vx, vy)

Usage:
    python3 eval/capture_ablation_data.py --out eval/data/ablation/a1 --duration 280
"""

import argparse
import os
import signal
import struct
import sys
import time
from pathlib import Path

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from nav_msgs.msg import OccupancyGrid, Odometry
    import math
except ImportError:
    print("ERROR: rclpy not available. Source /opt/ros/humble/setup.bash")
    sys.exit(1)


# ── euler from quaternion ──────────────────────────────────────────────────────
def quat_to_rpy(qx, qy, qz, qw):
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr, cosr)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny, cosy)

    return roll, pitch, yaw


class AblationCapture(Node):
    def __init__(self, out_dir: Path, duration: float):
        super().__init__("ablation_capture")
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.duration = duration
        self.start_time = time.time()

        self.last_map_msg = None
        self.odom_rows = []

        # QoS for map — match the publisher (create_publisher depth=10, default QoS)
        # The temporal_radar_occupancy_grid node publishes with default QoS (RELIABLE, VOLATILE)
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # Sensor QoS for odometry
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self._map_cb, map_qos)
        self.odom_sub = self.create_subscription(
            Odometry, "/fused_odom/odometry", self._odom_cb, odom_qos)

        self.timer = self.create_timer(1.0, self._tick)
        self.get_logger().info(
            f"AblationCapture started — saving to {out_dir}, duration={duration}s")

    def _map_cb(self, msg: OccupancyGrid):
        self.last_map_msg = msg
        self.get_logger().info(
            f"Map received: {msg.info.width}×{msg.info.height} @ {msg.info.resolution:.3f}m/cell",
            throttle_duration_sec=10.0)

    def _odom_cb(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        roll, pitch, yaw = quat_to_rpy(q.x, q.y, q.z, q.w)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.odom_rows.append([t, x, y, roll, pitch, yaw, vx, vy])

    def _tick(self):
        elapsed = time.time() - self.start_time
        n_odom = len(self.odom_rows)
        has_map = self.last_map_msg is not None
        self.get_logger().info(
            f"t={elapsed:.0f}/{self.duration:.0f}s  odom={n_odom}  map={'YES' if has_map else 'NO'}")
        if elapsed >= self.duration:
            self._save_and_shutdown()

    def _save_and_shutdown(self):
        self.get_logger().info("Duration reached — saving outputs...")
        self._save_map()
        self._save_odom()
        rclpy.shutdown()

    def _save_map(self):
        if self.last_map_msg is None:
            self.get_logger().warn("No map message received — map.pgm NOT saved")
            return

        msg = self.last_map_msg
        w, h = msg.info.width, msg.info.height
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y

        # Convert OccupancyGrid (-1, 0-100) → PGM (0=black=occupied, 205=grey=unknown, 254=white=free)
        data = np.array(msg.data, dtype=np.int8).reshape(h, w)
        pgm = np.full((h, w), 205, dtype=np.uint8)  # unknown
        pgm[data == 0]   = 254   # free → white
        pgm[data > 50]   = 0     # occupied → black
        # Flip vertically (ROS map origin is bottom-left, PGM top-left)
        pgm = np.flipud(pgm)

        pgm_path = self.out_dir / "map.pgm"
        with open(pgm_path, "wb") as f:
            # PGM P5 binary header
            header = f"P5\n{w} {h}\n255\n"
            f.write(header.encode())
            f.write(pgm.tobytes())

        # Save YAML metadata
        yaml_path = self.out_dir / "map_meta.yaml"
        yaml_path.write_text(
            f"image: map.pgm\n"
            f"resolution: {res}\n"
            f"origin: [{ox}, {oy}, 0.0]\n"
            f"negate: 0\n"
            f"occupied_thresh: 0.65\n"
            f"free_thresh: 0.196\n"
        )
        self.get_logger().info(f"Saved map: {pgm_path}  ({w}×{h})")

    def _save_odom(self):
        if not self.odom_rows:
            self.get_logger().warn("No odometry messages received — odom.npz NOT saved")
            return
        arr = np.array(self.odom_rows, dtype=np.float64)
        odom_path = self.out_dir / "odom.npz"
        np.savez_compressed(str(odom_path), data=arr)
        self.get_logger().info(
            f"Saved odometry: {odom_path}  ({len(self.odom_rows)} poses)")


def main():
    parser = argparse.ArgumentParser(description="Capture ablation map + odometry")
    parser.add_argument("--out",      required=True, help="Output directory")
    parser.add_argument("--duration", type=float, default=280.0,
                        help="How long to collect data before saving (seconds)")
    args = parser.parse_args()

    rclpy.init()
    node = AblationCapture(Path(args.out), args.duration)

    def _sigterm(sig, frame):
        node.get_logger().info("SIGTERM received — saving and shutting down")
        node._save_and_shutdown()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT,  _sigterm)

    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        node._save_map()
        node._save_odom()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
