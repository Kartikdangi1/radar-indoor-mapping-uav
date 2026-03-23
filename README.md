# Radar-Based Indoor Occupancy Mapping on a UAV

**Temporal filtering and system integration for 4D automotive radar on embedded platforms**

[![ROS 2 Humble](https://img.shields.io/badge/ROS%202-Humble-blue?logo=ros)](https://docs.ros.org/en/humble/)
[![Platform](https://img.shields.io/badge/Jetson-Orin%20NX-green?logo=nvidia)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)

*Kartik Dangi · March 2026*

</div>

---

<div align="center">
<img src="images/IsometricDrone+Radar+MediaConverter3Drender.jpeg" width="55%" alt="UAV + Radar assembly"/>

*UAV with 4D automotive radar + Ethernet media converter*
</div>

---

## What This Is

A ROS 2 system that turns a **4D automotive radar** into a real-time indoor mapping tool on a UAV — no GPS, no camera, no LiDAR map. Works in smoke, darkness, and GPS-denied environments.

### Key Features

| | |
|---|---|
| **Fused Odometry** | Madgwick IMU + RANSAC Doppler + LiDAR height → 50 Hz |
| **Radar SLAM** | GICP scan matching + loop closure for drift correction |
| **Temporal Occupancy Grid** | Bayesian log-odds with occlusion filtering + temporal decay |
| **Embedded Real-Time** | ~150 ms full pipeline latency on Jetson Orin NX |

Automotive radar operating at 77 GHz penetrates smoke, dust, and darkness — conditions that cause cameras and LiDARs to fail. It also provides per-point Doppler velocity, enabling ego-motion estimation without any visual features.

---

## System Architecture

<div align="center">
<img src="images/SystemArch.png" width="85%" alt="System architecture"/>
</div>

Two parallel preprocessing paths share the same radar hardware:

```
4D Radar ──┬──→ Odometry Path  (40° FoV, −20 dB RCS)  →  GICP SLAM + Loop Closure
           └──→ Mapping Path   (70° FoV,  −4 dB RCS)  →  Temporal Occupancy Grid
```

<div align="center">
<img src="images/ros_node_graph.png" width="85%" alt="ROS 2 node graph"/>

*Live ROS 2 node graph*
</div>

---

## SLAM Pipeline

<div align="center">
<img src="images/slam_pipeline.png" width="80%" alt="SLAM pipeline diagram"/>
</div>

---

## Results

### Room Experiment (40 m²)

<div align="center">

<img src="images/eval/fig_exp1_room_map.png" width="48%" alt="Room occupancy map"/>
<img src="images/eval/fig_exp1_map_with_trajectory.png" width="48%" alt="Room map with trajectory"/>

*Generated occupancy map (7 cm resolution) and flight trajectory overlay*
</div>

### Corridor Experiment (2 laps, ~25 m)

<div align="center">

<img src="images/eval/fig_exp3_corridor_map.png" width="48%" alt="Corridor map"/>
<img src="images/eval/fig_exp3_lap_comparison.png" width="48%" alt="Lap comparison"/>

*Corridor occupancy map and lap-over-lap drift analysis*
</div>

### Occlusion Filtering

<div align="center">
<img src="images/eval/fig_exp2_occlusion.png" width="75%" alt="Effect of occlusion filtering"/>

*Per-ray occlusion filtering removes phantom occupancy behind detected surfaces*
</div>

### Performance on Jetson Orin NX

<div align="center">

<img src="images/eval/fig5_8_performance_latency.png" width="48%" alt="Pipeline latency"/>
<img src="images/eval/fig5_9_cpu_ram.png" width="48%" alt="CPU and RAM"/>

*~150 ms end-to-end latency; sustained operation well within embedded hardware limits*
</div>

---

## Hardware

<div align="center">

<img src="images/Front-Drone.png" width="32%" alt="Drone front"/>
<img src="images/RadarMountFront.png" width="32%" alt="Radar mount front"/>
<img src="images/WiringDiagram.png" width="32%" alt="Wiring diagram"/>

*UAV platform · custom radar bracket · full wiring diagram*
</div>

---

## Quick Start

```bash
# Build
source /opt/ros/humble/setup.bash
colcon build --symlink-install && source install/setup.bash

# Run (with a bag file)
ros2 launch temporal_radar_mapping radar_mapping.launch.py use_sim_time:=true launch_rviz:=true
ros2 bag play /path/to/bag.mcap --clock

# Save map
ros2 service call /save_map std_srvs/srv/Trigger "{}"
```

**Prerequisites**: ROS 2 Humble, PCL, Eigen, tf2

---

## Packages

| Package | Role |
|---|---|
| `fused_odometry` | IMU + Doppler + height fusion, GICP SLAM |
| `temporal_radar_mapping` | Bayesian occupancy grid, map saver |
| `radar_messages` | Custom ROS 2 message definitions |

All config in `src/<package>/config/`. Default: 7 cm resolution, 4 s decay delay.

---

## Evaluation

```bash
./eval/09_run_experiments.sh   # full suite — ~50 min, generates all figures
python3 eval/02_plot_trajectory.py --data eval/data --out eval/figures
python3 eval/04_plot_performance.py --data eval/data --out eval/figures
python3 eval/08_enhance_map.py --map eval/maps/baseline.pgm --simple
```

---

*Kartik Dangi*
