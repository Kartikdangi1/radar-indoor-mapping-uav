#!/usr/bin/env python3
"""
01_extract_bag_data.py
======================
Offline extraction of key signals from an MCAP bag file.
Saves numpy / CSV files to eval/data/ for downstream plots.

Extracted topics:
  /fused_odom/odometry         → data/odom.npz   (t, x, y, roll, pitch, yaw, vx, vy)
  /robot/sensor/imu/data       → data/imu.npz    (t, ax, ay, az, wx, wy, wz)
  /PointCloudDetection         → data/radar_meta.npz (t, n_points per scan)
  /PointCloudDetectionFiltered → data/radar_odom_meta.npz
  /PointCloudDetectionMapping  → data/radar_map_meta.npz
  /fused_odom/loop_closures    → data/loop_closures.npz (stamp)

Usage (source ROS Humble first):
  source /opt/ros/humble/setup.bash
  python3 eval/01_extract_bag_data.py --bag /path/to/bag.mcap

Output: eval/data/*.npz (compressed numpy archives)
"""

import argparse
import os
import sys
import struct
import numpy as np

try:
    import rclpy
    from rclpy.serialization import deserialize_message
    import rosbag2_py
    from rosidl_runtime_py.utilities import get_message
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("[WARN] rclpy not found. Please source /opt/ros/humble/setup.bash")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def detect_storage_id(bag_path: str) -> str:
    """Detect storage format by inspecting files in the bag directory."""
    check = [bag_path] if os.path.isfile(bag_path) else \
            [os.path.join(bag_path, f) for f in os.listdir(bag_path)] if os.path.isdir(bag_path) else []
    for p in check:
        if p.endswith(".db3"):
            return "sqlite3"
        if p.endswith(".mcap"):
            return "mcap"
    return "mcap"  # default


def get_reader(bag_path: str):
    storage_id = detect_storage_id(bag_path)
    storage_opts = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
    conv_opts = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_opts, conv_opts)
    return reader


def topic_type_map(reader) -> dict:
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


# ---------------------------------------------------------------------------
# Per-topic extractors
# ---------------------------------------------------------------------------

def extract_odometry(reader, type_map: dict, topic: str):
    if topic not in type_map:
        print(f"  [SKIP] {topic} not in bag")
        return None
    msg_type = get_message(type_map[topic])
    rows = []
    while reader.has_next():
        _topic, data, _ts = reader.read_next()
        if _topic != topic:
            continue
        msg = deserialize_message(data, msg_type)
        t = stamp_to_sec(msg.header.stamp)
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        from scipy.spatial.transform import Rotation
        rpy = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_euler('xyz')
        rows.append([t, p.x, p.y, rpy[0], rpy[1], rpy[2], v.x, v.y])
    if not rows:
        return None
    arr = np.array(rows, dtype=np.float64)
    return arr  # columns: t, x, y, roll, pitch, yaw, vx, vy


def extract_imu(reader, type_map: dict, topic: str):
    if topic not in type_map:
        return None
    msg_type = get_message(type_map[topic])
    rows = []
    while reader.has_next():
        _topic, data, _ts = reader.read_next()
        if _topic != topic:
            continue
        msg = deserialize_message(data, msg_type)
        t = stamp_to_sec(msg.header.stamp)
        a = msg.linear_acceleration
        g = msg.angular_velocity
        rows.append([t, a.x, a.y, a.z, g.x, g.y, g.z])
    if not rows:
        return None
    return np.array(rows, dtype=np.float64)


def extract_range(reader, type_map: dict, topic: str):
    if topic not in type_map:
        return None
    msg_type = get_message(type_map[topic])
    rows = []
    while reader.has_next():
        _topic, data, _ts = reader.read_next()
        if _topic != topic:
            continue
        msg = deserialize_message(data, msg_type)
        t = stamp_to_sec(msg.header.stamp)
        rows.append([t, msg.range])
    if not rows:
        return None
    return np.array(rows, dtype=np.float64)


def extract_pointcloud_meta(reader, type_map: dict, topic: str):
    """Extract (timestamp, n_points) for each PointCloud2 message."""
    if topic not in type_map:
        return None
    msg_type = get_message(type_map[topic])
    rows = []
    while reader.has_next():
        _topic, data, _ts = reader.read_next()
        if _topic != topic:
            continue
        msg = deserialize_message(data, msg_type)
        t = stamp_to_sec(msg.header.stamp)
        n = msg.width * msg.height
        rows.append([t, float(n)])
    if not rows:
        return None
    return np.array(rows, dtype=np.float64)


def extract_loop_closure_stamps(reader, type_map: dict, topic: str = "/fused_odom/loop_closures"):
    """Extract timestamps of loop closure marker arrays (one per event)."""
    if topic not in type_map:
        return None
    msg_type = get_message(type_map[topic])
    stamps = []
    while reader.has_next():
        _topic, data, _ts = reader.read_next()
        if _topic != topic:
            continue
        msg = deserialize_message(data, msg_type)
        if msg.markers:
            t = stamp_to_sec(msg.markers[0].header.stamp)
            stamps.append(t)
    if not stamps:
        return None
    return np.array(stamps, dtype=np.float64)


# ---------------------------------------------------------------------------
# Multi-pass extraction (one pass per topic group to avoid seeking)
# ---------------------------------------------------------------------------

TOPICS = {
    "odom":          "/fused_odom/odometry",
    "imu":           "/robot/sensor/imu/data",
    "radar_raw":     "/PointCloudDetection",
    "radar_odom":    "/PointCloudDetectionFiltered",
    "radar_map":     "/PointCloudDetectionMapping",
    "loop_closures": "/fused_odom/loop_closures",
}


def extract_all(bag_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Single pass: collect everything together
    storage_opts = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
    conv_opts = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_opts, conv_opts)
    type_map = topic_type_map(reader)

    print(f"  Topics in bag: {list(type_map.keys())}")

    # Accumulators
    data: dict = {k: [] for k in TOPICS}
    msg_types = {}
    for key, topic in TOPICS.items():
        if topic in type_map:
            msg_types[topic] = get_message(type_map[topic])

    while reader.has_next():
        topic, raw, _ts = reader.read_next()
        if topic not in msg_types:
            continue
        msg = deserialize_message(raw, msg_types[topic])

        # Odometry
        if topic == TOPICS["odom"]:
            t = stamp_to_sec(msg.header.stamp)
            p = msg.pose.pose.position
            o = msg.pose.pose.orientation
            v = msg.twist.twist.linear
            from scipy.spatial.transform import Rotation
            rpy = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_euler('xyz')
            data["odom"].append([t, p.x, p.y, rpy[0], rpy[1], rpy[2], v.x, v.y])

        # IMU
        elif topic == TOPICS["imu"]:
            t = stamp_to_sec(msg.header.stamp)
            a = msg.linear_acceleration
            g = msg.angular_velocity
            data["imu"].append([t, a.x, a.y, a.z, g.x, g.y, g.z])

        # PointCloud metas
        elif topic in (TOPICS["radar_raw"], TOPICS["radar_odom"], TOPICS["radar_map"]):
            t = stamp_to_sec(msg.header.stamp)
            n = msg.width * msg.height
            key = {TOPICS["radar_raw"]: "radar_raw",
                   TOPICS["radar_odom"]: "radar_odom",
                   TOPICS["radar_map"]: "radar_map"}[topic]
            data[key].append([t, float(n)])

        # Loop closures
        elif topic == TOPICS["loop_closures"]:
            if msg.markers:
                t = stamp_to_sec(msg.markers[0].header.stamp)
                data["loop_closures"].append([t])

    # Save
    saved = []
    for key, rows in data.items():
        if rows:
            arr = np.array(rows, dtype=np.float64)
            path = os.path.join(out_dir, f"{key}.npz")
            np.savez_compressed(path, data=arr)
            saved.append(f"    {key}: {arr.shape[0]} samples → {path}")
        else:
            saved.append(f"    {key}: 0 samples (topic not in bag or no data)")

    for line in saved:
        print(line)

    # Also save a summary JSON
    import json
    summary = {k: (len(rows) if rows else 0) for k, rows in data.items()}
    summary_path = os.path.join(out_dir, "extraction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract bag data for thesis evaluation")
    parser.add_argument("--bag", required=True, help="Path to .mcap bag file or bag directory")
    parser.add_argument("--out", default="eval/data", help="Output directory for .npz files")
    args = parser.parse_args()

    if not os.path.exists(args.bag):
        print(f"ERROR: Bag not found: {args.bag}")
        sys.exit(1)

    print(f"Extracting: {args.bag}")
    print(f"Output dir: {args.out}")
    extract_all(args.bag, args.out)
    print("\nDone. Run 02_plot_trajectory.py next.")


if __name__ == "__main__":
    main()
