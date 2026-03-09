# Radar-Based Indoor Occupancy Mapping on a UAV

Temporal filtering and system integration for 4D automotive radar on embedded platforms.

**Author**: Kartik Dangi
**Institution**: Technische Hochschule Würzburg-Schweinfurt (THWS)
**Degree**: Bachelor of Engineering
**Submission**: March 14, 2026

## Overview

This repository contains a ROS 2 (Humble) system for real-time indoor occupancy grid mapping using a 4D automotive radar mounted on a UAV. The system fuses radar measurements with inertial and height data to enable robust, temporally-filtered 2D maps in GPS-denied indoor environments.

### Key Features

- **Fused odometry**: Madgwick IMU orientation + RANSAC Doppler velocity + LiDAR height fusion (50 Hz)
- **Radar SLAM**: GICP-based scan matching with loop closure detection for drift correction
- **Temporal occupancy grid**: Bayesian log-odds with per-ray occlusion filtering and temporal decay
- **Indoor-optimized sensor model**: Dynamic RCS thresholds, multipath rejection, motion compensation
- **Real-time performance**: Operates on embedded GPU (Jetson Orin) at full pipeline latency ~150 ms

## System Architecture

Two independent preprocessing pipelines:

```
4D Automotive Radar Input
├─→ Odometry Path     (40° FoV, -20dB RCS)
│   ├→ Fused Odometry (IMU+Doppler+Height)
│   └→ Scan SLAM (GICP + Loop Closure)
│
└─→ Mapping Path      (70° FoV, -4dB RCS)
    └→ Temporal Occupancy Grid (Bayesian + Decay)
```

**Transform Tree**: `map → odom → base_link → {imu, radar}`

## Quick Start

### Prerequisites

- ROS 2 Humble (Ubuntu 22.04)
- colcon-core
- PCL, Eigen, tf2

### Building

```bash
source /opt/ros/humble/setup.bash
cd radar-indoor-mapping-uav
colcon build --symlink-install
source install/setup.bash
```

### Running with a Dataset

```bash
# Terminal 1: Launch full pipeline
ros2 launch temporal_radar_mapping radar_mapping.launch.py use_sim_time:=true launch_rviz:=true

# Terminal 2: Play bag file
ros2 bag play /path/to/bag.mcap --clock
```

### Saving Maps

```bash
ros2 run temporal_radar_mapping map_saver.py \
    --ros-args -p use_sim_time:=true \
    -p "output_dir:=$HOME/maps" \
    -p "map_name:=my_map"

# Trigger save when ready
ros2 service call /save_map std_srvs/srv/Trigger "{}"
```

## Evaluation

All evaluation scripts are in `eval/` with a shared config file `eval_config.sh`:

### Full Experiment Suite (All Datasets + Figures)

```bash
source install/setup.bash
./eval/09_run_experiments.sh
```

Runs 5 experiments sequentially (~50 min total):
1. Room dataset, all features ON
2. Room dataset, occlusion filtering OFF
3. Corridor full trajectory
4. Corridor lap 1 only
5. Corridor lap 2 only

Generates all figures, metrics, and thesis PDFs automatically.

### Individual Evaluation Commands

```bash
# Extract odometry + radar from bag
python3 eval/01_extract_bag_data.py --bag <bag.mcap> --out eval/data

# Plot trajectory, velocity, orientation
python3 eval/02_plot_trajectory.py --data eval/data --out eval/figures

# Performance metrics (CPU/RAM/latency)
python3 eval/04_plot_performance.py --data eval/data --out eval/figures

# Map quality analysis
python3 eval/05_analyze_map.py --map eval/maps/baseline.pgm --out eval/figures

# Map enhancement (filtering, post-processing)
python3 eval/08_enhance_map.py --map eval/maps/baseline.pgm --simple

# Generate all thesis figures
python3 eval/07_generate_thesis_figures.py
```

See individual scripts for full argument lists and options.

## Packages

### fused_odometry

Sensor fusion and odometry estimation:
- `fused_odom_node` — Sensor fusion (IMU + Doppler + Height)
- `radar_preprocessing_node` — Point cloud filtering (odometry + mapping paths)
- `radar_scan_slam_node` — GICP-based SLAM with loop closure detection

**Config**: `src/fused_odometry/config/`

### temporal_radar_mapping

Occupancy grid mapping:
- `temporal_radar_occupancy_node` — Bayesian log-odds grid with temporal decay
- `map_saver.py` — Saves grids to PGM format

**Config**: `src/temporal_radar_mapping/config/`

### radar_messages

Custom ROS message definitions for radar sensor interface.

## Configuration

All parameters are in YAML files under `src/<package>/config/`. Key configs:

| Config File | Purpose |
|------------|---------|
| `fused_odom_config.yaml` | IMU, Doppler, height fusion tuning |
| `radar_preprocessing_node.yaml` | Odometry-path filtering |
| `radar_preprocessing_mapping.yaml` | Mapping-path filtering (wider FoV, lower RCS) |
| `radar_scan_slam.yaml` | GICP, loop closure parameters |
| `temporal_radar_occupancy_config.yaml` | Grid resolution, decay, sensor model |

Default: 7cm grid resolution, 4s decay delay, multipath rejection enabled.

## Debugging

### Common Issues

**No map appearing after 10+ seconds**:
```bash
ros2 topic hz /fused_odom/odometry  # Should be ~50 Hz
```
If not publishing, fused odometry is warming up (IMU gyro calibration, ~10s). This is normal.

**Map drifting significantly**:
- Check `use_sim_time` consistency across all nodes
- Verify motion compensation is enabled: `enable_motion_compensation: true`
- Lower RANSAC thresholds if Doppler inliers are few

**Loop closures not detected**:
- Verify `enable_loop_closure: true` in `radar_scan_slam.yaml`
- Lower `loop_closure_fitness_threshold` (try 0.5)
- Lower `loop_closure_min_distance` if revisiting sooner than 5m

## Results

### Room Environment (40 m²)
- Map resolution: 7 cm
- Occupancy grid size: 200 m²
- Processing time: ~150 ms per frame
- Accuracy: < 5% cell disagreement vs. ground truth

### Corridor Environment (2 laps, ~25 m)
- Dual-lap drift analysis available via `eval/02_plot_trajectory.py --lap-split <timestamp>`
- Loop closure correctness: ±0.2 m translation, ±0.05 rad rotation

## Project Structure

```
.
├── src/
│   ├── fused_odometry/              # Sensor fusion + SLAM
│   ├── temporal_radar_mapping/      # Occupancy grid
│   └── radar_messages/              # Custom message definitions
├── eval/
│   ├── 01_extract_bag_data.py       # Extract odometry + radar
│   ├── 02_plot_trajectory.py        # Trajectory/velocity plots
│   ├── 03_collect_live_performance.py
│   ├── 04_plot_performance.py       # CPU/RAM/latency metrics
│   ├── 05_analyze_map.py            # Map quality
│   ├── 06_run_ablation.sh           # Ablation study
│   ├── 07_generate_thesis_figures.py # All thesis PDFs
│   ├── 08_enhance_map.py            # Post-processing
│   ├── 09_run_experiments.sh        # Full experiment suite
│   ├── 10_generate_experiment_figures.py
│   ├── eval_config.sh               # Shared config
│   └── launch_eval.sh               # Interactive launcher
└── README.md                         # This file
```

## Dependencies

- **ROS 2 Humble**: Core middleware
- **PCL 1.12**: Point cloud processing
- **Eigen 3**: Linear algebra
- **tf2**: Transform management
- **scipy, numpy, matplotlib**: Evaluation scripts

All packaged via `colcon build`. No additional system installs beyond ROS.

## Citation

If you use this system in your research, please cite:

```bibtex
@bachelorsthesis{dangi2026radar,
  author  = {Dangi, Kartik},
  title   = {Radar-Based Indoor Occupancy Mapping on a UAV: System Integration
             and Temporal Filtering for 4D Automotive Radar},
  school  = {Technische Hochschule W{\"u}rzburg-Schweinfurt},
  year    = {2026},
  month   = {March}
}
```

## References

Key algorithms and sensor models implemented:

- **Madgwick Filter** — IMU orientation estimation (NED convention)
- **RANSAC** — Robust Doppler velocity estimation
- **GICP** — Generalized ICP for scan registration
- **Bayesian Occupancy Grid** — Log-odds update with temporal decay
- **Occlusion Filtering** — Per-ray sensor model with angle-dependent RCS thresholds

Detailed references and derivations in thesis Chapter 3 (Methodology).

## License

This project is part of a bachelor thesis at THWS. All code is provided as-is for educational and research purposes.

## Contact

Kartik Dangi
Department of Engineering
Technische Hochschule Würzburg-Schweinfurt (THWS)
Würzburg, Germany
