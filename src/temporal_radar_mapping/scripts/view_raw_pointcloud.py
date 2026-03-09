#!/usr/bin/env python3
"""
View Raw Radar Point Cloud (ARS548) in RViz2
=============================================

Simple script to visualize raw /PointCloudDetection with a grey background.
Use this to capture Fig 3.8: Raw Radar in Foxglove/RViz.

USAGE:
  1. In terminal 1, play the rosbag:
     ros2 bag play rosbag2_2026_02_18-17_25_30_0.mcap --clock

  2. In terminal 2, run this script:
     python3 view_raw_pointcloud.py

  3. In RViz2, take a screenshot when you see points on the grey background.

REQUIREMENTS:
  - ROS2 Humble installed
  - temporal_radar_mapping package built
  - rosbag file available

DATA TO NOTE:
  Watch the RViz status bar for:
    - Number of points in current scan
    - Display FPS (should be 30)

  Or use:
    ros2 topic echo --once /PointCloudDetection | grep width
"""

import subprocess
import sys
import os

def main():
    # Get the package share path
    pkg_share_cmd = "ros2 pkg prefix temporal_radar_mapping"
    result = subprocess.run(pkg_share_cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ ERROR: Could not find temporal_radar_mapping package")
        print("   Make sure the package is built:")
        print("   cd /home/kartik/Thesis/fradar/radar-indoor-mapping-uav")
        print("   colcon build --symlink-install --packages-select temporal_radar_mapping")
        return 1

    pkg_path = result.stdout.strip()
    rviz_config = os.path.join(pkg_path, "share", "temporal_radar_mapping", "rviz", "raw_pointcloud.rviz")

    if not os.path.exists(rviz_config):
        print(f"❌ ERROR: RViz config not found at {rviz_config}")
        return 1

    print("=" * 70)
    print("RAW POINT CLOUD VIEWER (Fig 3.8)")
    print("=" * 70)
    print(f"✓ RViz config: {rviz_config}")
    print()
    print("📌 WORKFLOW:")
    print("  1. Terminal 1: ros2 bag play rosbag2_2026_02_18-17_25_30_0.mcap --clock")
    print("  2. Terminal 2: python3 view_raw_pointcloud.py  (this script)")
    print("  3. RViz will open showing raw point cloud on grey background")
    print("  4. Take screenshot when you see points")
    print()
    print("💡 TIPS:")
    print("  - Grey background makes points stand out")
    print("  - Rainbow colormap shows intensity (RCS)")
    print("  - Check RViz status bar for point count")
    print("  - Use scroll wheel to zoom in/out")
    print("  - Right-click drag to pan")
    print("=" * 70)
    print()

    # Launch RViz2
    print("🚀 Launching RViz2...")
    rviz_cmd = f"rviz2 -d {rviz_config}"
    subprocess.run(rviz_cmd, shell=True)

    return 0

if __name__ == "__main__":
    sys.exit(main())
