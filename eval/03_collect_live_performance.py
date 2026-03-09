#!/usr/bin/env python3
"""
03_collect_live_performance.py
==============================
ROS2 node that measures processing latency and throughput of each pipeline node
DURING a live run (bag playback or real hardware).

Approach: measure wall-clock latency between input message receipt and
corresponding output message receipt for each node stage.

Stages measured:
  A. radar_preprocessing_node (odom path):
       /PointCloudDetection  → /PointCloudDetectionFiltered
  B. radar_preprocessing_mapping_node (map path):
       /PointCloudDetection  → /PointCloudDetectionMapping
  C. fused_odom_node:
       /PointCloudDetectionFiltered  → /fused_odom/odometry
  D. temporal_radar_occupancy_node:
       /PointCloudDetectionMapping + /fused_odom/odometry → /map
  E. System CPU + RAM (sampled every 1 s via psutil)

Writes results to eval/data/performance_metrics.json

Usage:
    # Terminal 1: start the pipeline
    ros2 launch temporal_radar_mapping radar_mapping.launch.py use_sim_time:=true

    # Terminal 2: start the bag
    ros2 bag play /path/to/bag.mcap --clock

    # Terminal 3: run this node (source ROS first)
    python3 eval/03_collect_live_performance.py --duration 120

    # After bag finishes:
    python3 eval/04_plot_performance.py
"""

import argparse
import json
import os
import sys
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import OccupancyGrid, Odometry
    from visualization_msgs.msg import MarkerArray
    HAS_ROS = True
except ImportError:
    print("ERROR: rclpy not available. Source /opt/ros/humble/setup.bash")
    sys.exit(1)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    print("[WARN] psutil not installed. CPU/RAM monitoring disabled. Install with: pip install psutil")
    HAS_PSUTIL = False


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LatencyRecord:
    """Wall-clock time (seconds) between receiving input and seeing output."""
    stage: str
    latencies_ms: list = field(default_factory=list)

    def record(self, dt_s: float):
        self.latencies_ms.append(dt_s * 1000.0)

    def stats(self) -> dict:
        if not self.latencies_ms:
            return {"count": 0}
        arr = sorted(self.latencies_ms)
        import statistics
        return {
            "count":    len(arr),
            "mean_ms":  statistics.mean(arr),
            "median_ms": statistics.median(arr),
            "std_ms":   statistics.stdev(arr) if len(arr) > 1 else 0.0,
            "p95_ms":   arr[int(0.95 * len(arr))],
            "p99_ms":   arr[int(0.99 * len(arr))],
            "min_ms":   arr[0],
            "max_ms":   arr[-1],
        }


@dataclass
class FrequencyRecord:
    """Rolling message rate tracker."""
    topic: str
    timestamps: deque = field(default_factory=lambda: deque(maxlen=500))

    def record(self):
        self.timestamps.append(time.monotonic())

    def hz(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        window = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / max(window, 1e-6)


# ---------------------------------------------------------------------------
# Monitor node
# ---------------------------------------------------------------------------

class PerformanceMonitor(Node):
    def __init__(self, output_path: str, duration: float):
        super().__init__("performance_monitor")

        self.output_path_ = output_path
        self.start_time_ = time.monotonic()
        self.duration_ = duration

        # Latency records per stage
        self.lat_preproc_odom_  = LatencyRecord("preproc_odom")
        self.lat_preproc_map_   = LatencyRecord("preproc_map")
        self.lat_fused_odom_    = LatencyRecord("fused_odom")
        self.lat_occupancy_     = LatencyRecord("occupancy_grid")

        # Frequency records
        self.freq: dict[str, FrequencyRecord] = {
            t: FrequencyRecord(t) for t in [
                "/PointCloudDetection",
                "/PointCloudDetectionFiltered",
                "/PointCloudDetectionMapping",
                "/fused_odom/odometry",
                "/map",
            ]
        }

        # Latest input timestamps (wall clock) keyed by header stamp (ns int)
        self._radar_in_times: dict[int, float] = {}     # stamp_ns → wall_clock
        self._odom_filter_times: dict[int, float] = {}  # stamp_ns → wall_clock
        self._map_filter_times: dict[int, float] = {}   # stamp_ns → wall_clock

        # ------------------------------------------------------------------
        # Live signal recording (saved to NPZ for offline plots)
        # ------------------------------------------------------------------
        # Odometry: [t, x, y, z, vx, vy, vz]
        self._odom_rows: list = []
        # Radar cloud metadata: [t, n_points]
        self._radar_raw_rows: list = []
        self._radar_odom_rows: list = []
        self._radar_map_rows: list = []
        # Loop closure event timestamps
        self._loop_closure_stamps: list = []

        # Fused-odom latency: pair latest radar-in wall time with odom arrival
        # (approximate — fused odom runs at 50 Hz fusing multiple radar scans)
        self._last_radar_in_wall: Optional[float] = None

        # CPU/RAM log
        self.cpu_ram_log_: list = []

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        # Subscriptions
        self.sub_radar_in_    = self.create_subscription(
            PointCloud2, "/PointCloudDetection",
            self._cb_radar_in, sensor_qos)
        self.sub_radar_odom_  = self.create_subscription(
            PointCloud2, "/PointCloudDetectionFiltered",
            self._cb_radar_odom_out, sensor_qos)
        self.sub_radar_map_   = self.create_subscription(
            PointCloud2, "/PointCloudDetectionMapping",
            self._cb_radar_map_out, sensor_qos)
        self.sub_odom_        = self.create_subscription(
            Odometry, "/fused_odom/odometry",
            self._cb_odom_out, sensor_qos)
        self.sub_grid_        = self.create_subscription(
            OccupancyGrid, "/map",
            self._cb_grid_out, sensor_qos)
        self.sub_loop_        = self.create_subscription(
            MarkerArray, "/fused_odom/loop_closures",
            self._cb_loop_closure, sensor_qos)

        # CPU/RAM timer (every 1 s)
        if HAS_PSUTIL:
            self._proc = psutil.Process()
            self.create_timer(1.0, self._sample_system)

        # Shutdown timer
        self.create_timer(min(5.0, duration), self._check_shutdown)

        self.get_logger().info(
            f"Performance monitor started (duration={duration:.0f}s). "
            f"Output: {output_path}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _stamp_ns(self, header) -> int:
        return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec

    def _cb_radar_in(self, msg: PointCloud2):
        now = time.monotonic()
        ns = self._stamp_ns(msg.header)
        self._radar_in_times[ns] = now
        self._last_radar_in_wall = now
        if len(self._radar_in_times) > 500:
            oldest = min(self._radar_in_times)
            del self._radar_in_times[oldest]
        self.freq["/PointCloudDetection"].record()
        # Record point count for radar stats
        t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        n = float(msg.width * msg.height)
        self._radar_raw_rows.append([t_sec, n])

    def _cb_radar_odom_out(self, msg: PointCloud2):
        now = time.monotonic()
        ns = self._stamp_ns(msg.header)
        self.freq["/PointCloudDetectionFiltered"].record()
        self._odom_filter_times[ns] = now
        if len(self._odom_filter_times) > 500:
            oldest = min(self._odom_filter_times)
            del self._odom_filter_times[oldest]
        # Preprocessing latency (odom path): match by exact stamp
        if ns in self._radar_in_times:
            dt = now - self._radar_in_times[ns]
            self.lat_preproc_odom_.record(dt)
        t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._radar_odom_rows.append([t_sec, float(msg.width * msg.height)])

    def _cb_radar_map_out(self, msg: PointCloud2):
        now = time.monotonic()
        ns = self._stamp_ns(msg.header)
        self.freq["/PointCloudDetectionMapping"].record()
        self._map_filter_times[ns] = now
        if len(self._map_filter_times) > 500:
            oldest = min(self._map_filter_times)
            del self._map_filter_times[oldest]
        if ns in self._radar_in_times:
            dt = now - self._radar_in_times[ns]
            self.lat_preproc_map_.record(dt)
        t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._radar_map_rows.append([t_sec, float(msg.width * msg.height)])

    def _cb_odom_out(self, msg: Odometry):
        now = time.monotonic()
        self.freq["/fused_odom/odometry"].record()

        # Fused-odom latency: use most recently seen radar input as reference.
        # Fused odom runs at 50 Hz driven by IMU, incorporating the latest radar
        # Doppler — the wall-clock gap between last radar arrival and this odom
        # output approximates the radar→odom processing delay.
        if self._last_radar_in_wall is not None:
            dt = now - self._last_radar_in_wall
            if 0.0 < dt < 0.5:   # sanity: reject stale references > 500 ms
                self.lat_fused_odom_.record(dt)

        # Record full odometry for trajectory plots
        t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        try:
            from scipy.spatial.transform import Rotation
            rpy = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_euler('xyz')
        except Exception:
            rpy = [0.0, 0.0, 0.0]
        self._odom_rows.append([t_sec, p.x, p.y, rpy[0], rpy[1], rpy[2], v.x, v.y])

    def _cb_grid_out(self, msg: OccupancyGrid):
        now = time.monotonic()
        ns = self._stamp_ns(msg.header)
        self.freq["/map"].record()
        if self._map_filter_times:
            latest_ns = max(self._map_filter_times)
            if latest_ns <= ns:
                dt = now - self._map_filter_times[latest_ns]
                self.lat_occupancy_.record(dt)

    def _cb_loop_closure(self, msg: MarkerArray):
        if msg.markers:
            t_sec = msg.markers[0].header.stamp.sec + \
                    msg.markers[0].header.stamp.nanosec * 1e-9
            self._loop_closure_stamps.append([t_sec])

    def _sample_system(self):
        try:
            cpu_pct = psutil.cpu_percent(interval=None)
            ram_mb  = self._proc.memory_info().rss / (1024**2)
            self.cpu_ram_log_.append({
                "t": time.monotonic() - self.start_time_,
                "cpu_percent": cpu_pct,
                "ram_mb": ram_mb,
            })
        except Exception:
            pass

    def _check_shutdown(self):
        elapsed = time.monotonic() - self.start_time_
        if elapsed >= self.duration_:
            self.get_logger().info("Duration reached — saving results and shutting down.")
            self._save_results()
            rclpy.shutdown()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_results(self):
        results = {
            "latency": {
                "preproc_odom":   self.lat_preproc_odom_.stats(),
                "preproc_map":    self.lat_preproc_map_.stats(),
                "fused_odom":     self.lat_fused_odom_.stats(),
                "occupancy_grid": self.lat_occupancy_.stats(),
            },
            "frequency_hz": {
                topic: rec.hz() for topic, rec in self.freq.items()
            },
            "cpu_ram": self.cpu_ram_log_,
        }

        # Print table
        print("\n=== PERFORMANCE RESULTS ===")
        print(f"{'Stage':<30} {'Count':>6} {'Mean(ms)':>10} {'Std(ms)':>9} {'P95(ms)':>9}")
        print("-" * 68)
        for stage, s in results["latency"].items():
            if s.get("count", 0) > 0:
                print(f"{stage:<30} {s['count']:>6} {s['mean_ms']:>10.2f} "
                      f"{s['std_ms']:>9.2f} {s['p95_ms']:>9.2f}")

        print("\n=== TOPIC FREQUENCIES ===")
        for topic, hz in results["frequency_hz"].items():
            print(f"  {topic:<45} {hz:.1f} Hz")

        if self.cpu_ram_log_:
            cpus = [r["cpu_percent"] for r in self.cpu_ram_log_]
            rams = [r["ram_mb"] for r in self.cpu_ram_log_]
            print(f"\n  CPU:  mean={sum(cpus)/len(cpus):.1f}%  max={max(cpus):.1f}%")
            print(f"  RAM:  mean={sum(rams)/len(rams):.0f} MB  max={max(rams):.0f} MB")
            results["cpu_summary"] = {
                "mean_cpu_percent": sum(cpus) / len(cpus),
                "max_cpu_percent":  max(cpus),
                "mean_ram_mb":      sum(rams) / len(rams),
                "max_ram_mb":       max(rams),
            }

        out_dir = os.path.dirname(self.output_path_) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(self.output_path_, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {self.output_path_}")

        # ------------------------------------------------------------------
        # Save live-recorded signal data as NPZ (used by 02_plot_trajectory.py)
        # ------------------------------------------------------------------
        if HAS_NUMPY:
            npz_specs = [
                ("odom",        self._odom_rows),
                ("radar_raw",   self._radar_raw_rows),
                ("radar_odom",  self._radar_odom_rows),
                ("radar_map",   self._radar_map_rows),
                ("loop_closures", self._loop_closure_stamps),
            ]
            print("\n=== SIGNAL DATA (live-recorded) ===")
            for name, rows in npz_specs:
                if rows:
                    arr = np.array(rows, dtype=np.float64)
                    path = os.path.join(out_dir, f"{name}.npz")
                    np.savez_compressed(path, data=arr)
                    print(f"  {name:<20} {arr.shape[0]:>6} samples → {path}")
                else:
                    print(f"  {name:<20}      0 samples (topic not received)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=120.0,
                        help="Seconds to collect data (default: 120)")
    parser.add_argument("--out", default="eval/data/performance_metrics.json",
                        help="Output JSON path")
    args = parser.parse_args()

    rclpy.init()
    node = PerformanceMonitor(args.out, args.duration)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Always save — whether self-expired, SIGINT, or SIGTERM
        node._save_results()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
