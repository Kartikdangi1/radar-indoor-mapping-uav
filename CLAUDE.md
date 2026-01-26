# CLAUDE.md - Radar Indoor Mapping UAV

This file provides guidance for AI assistants working with this codebase.

## Project Overview

This is a **ROS 2-based radar SLAM and indoor mapping system** designed for UAVs equipped with ARS_548 automotive radar. The system provides:

- Real-time ego-velocity estimation from radar Doppler measurements
- Graph-based SLAM with Ceres optimization
- Temporal occupancy grid mapping with persistence

**Primary Use Case**: Autonomous indoor navigation and mapping using radar sensors, which excel in environments where LiDAR/cameras struggle (dust, smoke, low-light).

## Repository Structure

```
radar-indoor-mapping-uav/
├── CLAUDE.md                           # This file
├── .gitignore                          # Build artifacts, IDE files
└── src/
    ├── radar_odom/                     # Package 1: Odometry & SLAM
    │   ├── config/
    │   │   ├── graph_slam.yaml         # SLAM parameters
    │   │   └── radar_pcl_processor.yaml # Radar processing config
    │   ├── include/
    │   │   ├── RadarEgoVel.hpp         # Ego velocity estimation (RANSAC)
    │   │   ├── RadarEgoVelConfig.hpp   # Config struct
    │   │   ├── keyframe.h              # Keyframe class for SLAM
    │   │   ├── utils.hpp               # Transform utilities
    │   │   ├── ceres/                  # Ceres cost functions
    │   │   │   ├── CeresGraph.hpp
    │   │   │   ├── CostFunctionImu.h
    │   │   │   ├── CostFunctionOdom.h
    │   │   │   ├── CostFunctionOri.h
    │   │   │   └── CostFunctionTrans.h
    │   │   └── rio_utils/
    │   │       └── radar_point_cloud.hpp
    │   ├── src/
    │   │   ├── radar_pcl_processor.cpp # Main radar processing node (553 lines)
    │   │   ├── optimizer.cpp           # Graph SLAM node (533 lines)
    │   │   ├── baselink_tf.cpp         # TF broadcaster
    │   │   └── record.cpp              # Data recording
    │   ├── launch/
    │   │   ├── run.py                  # Full pipeline launch
    │   │   └── radar_mapping_full.py
    │   └── package.xml
    │
    └── temporal_radar_mapping/         # Package 2: Occupancy Grid Mapping
        ├── config/
        │   └── temporal_radar_occupancy_config.yaml
        ├── src/
        │   └── temporal_radar_occupancy_grid.cpp (722 lines)
        ├── launch/
        │   └── filter.launch.py
        ├── rviz/
        │   └── radar_mapping.rviz
        └── package.xml
```

## Key Technologies & Dependencies

### Build System
- **ROS 2** (tested with Humble/Iron)
- **ament_cmake** build tool
- **colcon** for workspace building

### Core Dependencies
```yaml
# From package.xml files
- rclcpp               # ROS 2 C++ client
- sensor_msgs          # PointCloud2, IMU
- nav_msgs             # Odometry, OccupancyGrid
- geometry_msgs        # Transforms, Twists
- tf2_ros              # Transform library
- pcl_ros              # Point Cloud Library
- Eigen3               # Linear algebra
- ceres-solver         # Graph optimization (radar_odom only)
- OpenCV               # Image operations (radar_pcl_processor)
```

### Hardware Target
- **ARS_548 Radar**: Automotive 4D radar with Doppler measurements
- **IMU**: For orientation estimation (Madgwick filter if no built-in orientation)

## Build Commands

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash  # or iron/rolling

# Build the workspace
cd /home/user/radar-indoor-mapping-uav
colcon build --symlink-install

# Source the workspace
source install/setup.bash
```

## Running the System

### Full SLAM Pipeline
```bash
# Launch odometry + SLAM + TF
ros2 launch radar_odom run.py
```

### Occupancy Grid Mapping Only
```bash
# Launch temporal mapping node
ros2 launch temporal_radar_mapping filter.launch.py
```

### With Custom Parameters
```bash
ros2 launch temporal_radar_mapping filter.launch.py \
    input_topic:=/my_radar/pointcloud \
    launch_rviz:=true
```

## Architecture & Data Flow

```
ARS_548 Radar (/PointCloudDetection)
        │
        ▼
┌─────────────────────────────────────┐
│   radar_pcl_processor (radar_odom)  │
│   ├─ Madgwick IMU filter            │
│   ├─ RANSAC ego-velocity            │
│   └─ Dynamic object removal         │
└─────────────────────────────────────┘
        │ /Ego_Vel_Twist
        ▼
┌─────────────────────────────────────┐
│      optimizer (radar_odom)         │
│   ├─ Keyframe creation              │
│   ├─ NDT/GICP registration          │
│   └─ Ceres pose graph               │
└─────────────────────────────────────┘
        │ /odometry, TF: map→odom→base_link
        ▼
┌─────────────────────────────────────┐
│ temporal_radar_occupancy_grid       │
│   ├─ TF-based transformation        │
│   ├─ Bayesian log-odds update       │
│   ├─ Temporal decay                 │
│   └─ Static object protection       │
└─────────────────────────────────────┘
        │ /map (OccupancyGrid)
        ▼
     RViz / Navigation Stack
```

## TF Frame Tree

```
map → odom → base_link → imu_link → ARS_548
```

Static transforms are published by the launch files. Adjust mounting offsets in `launch/run.py`.

## Key Algorithms

### 1. Ego Velocity Estimation (RadarEgoVel.hpp)
- RANSAC-based fitting of radar Doppler measurements
- Supports holonomic and non-holonomic vehicles
- Outputs 3D velocity with covariance

### 2. Graph SLAM (optimizer.cpp)
- Keyframe-based with configurable thresholds (0.5m, 15°)
- Multi-method registration: NDT + GICP
- Ceres solver with IMU/odometry constraints
- Point cloud accumulation for sparse radar

### 3. Occupancy Grid (temporal_radar_occupancy_grid.cpp)
- Bayesian log-odds model
- Temporal decay with static object protection
- RCS-weighted confidence
- Ray casting for free-space marking

## Configuration Files

### graph_slam.yaml
```yaml
# Key parameters
keyframe_delta_trans: 0.5      # meters
keyframe_delta_angle: 0.26     # radians (~15°)
max_icp_fitness: 8.0           # permissive for radar
accumulation_frames: 5         # frames to accumulate
```

### radar_pcl_processor.yaml
```yaml
# Key parameters
radar_topic: "/PointCloudDetection"
holonomic_vehicle: true
distance_near_thresh: 0.8      # meters
distance_far_thresh: 60.0      # meters
```

### temporal_radar_occupancy_config.yaml
```yaml
# Key parameters
grid_resolution: 0.25          # meters/cell
grid_size_m: 50.0              # 50x50m map
decay_factor: 0.995            # slow temporal decay
p_occ: 0.70                    # occupied probability
```

## Coding Conventions

### C++ Style
- **C++17** standard
- **ROS 2 naming**: snake_case for variables, CamelCase for classes
- Member variables suffixed with `_` (e.g., `position_`, `config_`)
- Use `EIGEN_MAKE_ALIGNED_OPERATOR_NEW` for Eigen members
- Namespaces: `rio` for radar utilities

### File Organization
- Headers in `include/` with `.hpp` or `.h`
- Implementation in `src/` with `.cpp`
- Configuration in `config/` with `.yaml`
- Launch files in `launch/` with `.py`

### ROS 2 Patterns
- Node classes inherit from `rclcpp::Node`
- Parameters declared in constructor
- Use smart pointers (`std::shared_ptr`, `::Ptr` for PCL)
- TF2 for all transform operations

## Common Development Tasks

### Adding a New Parameter
1. Add to appropriate `.yaml` config file
2. Declare in node constructor: `declare_parameter<type>("name", default)`
3. Read value: `get_parameter("name").as_type()`

### Modifying Registration
- Edit `optimizer.cpp`
- Registration settings in `graph_slam.yaml`
- Key functions: `registerPointCloud()`, `createKeyframe()`

### Adjusting Occupancy Grid Behavior
- Edit `temporal_radar_occupancy_grid.cpp`
- Parameters in `temporal_radar_occupancy_config.yaml`
- Key tunables: `p_occ`, `decay_factor`, `min_observations_for_static`

### Testing with Bag Files
```bash
# Play a bag file with radar data
ros2 bag play /path/to/bag --clock

# Launch with sim time
ros2 launch radar_odom run.py use_sim_time:=true
```

## Debugging Tips

### TF Issues
```bash
# View TF tree
ros2 run tf2_tools view_frames

# Echo specific transform
ros2 run tf2_ros tf2_echo map base_link
```

### Point Cloud Issues
```bash
# Echo point cloud info
ros2 topic echo /PointCloudDetection --field header

# Check field names
ros2 topic echo /PointCloudDetection --field fields
```

### SLAM Not Registering
1. Check `max_icp_fitness` - increase if failing frequently
2. Verify point counts with topic echo
3. Enable NDT if only using GICP
4. Increase `accumulation_frames`

## Topics Reference

| Topic | Type | Description |
|-------|------|-------------|
| `/PointCloudDetection` | PointCloud2 | Raw radar input |
| `/Ego_Vel_Twist` | TwistWithCovarianceStamped | Estimated velocity |
| `/odometry` | Odometry | SLAM odometry output |
| `/trajectory` | Path | Full trajectory |
| `/map` | OccupancyGrid | Occupancy grid map |

## Important Notes for AI Assistants

1. **This is a ROS 2 project** - ensure any suggestions use ROS 2 APIs, not ROS 1
2. **ARS_548 radar format** - RCS is UINT8 (0-255), not dB float
3. **Sparse point clouds** - radar produces far fewer points than LiDAR; parameters are tuned for this
4. **No CMakeLists.txt in repo** - these need to be created if building from scratch
5. **Two separate packages** - `radar_odom` and `temporal_radar_mapping` can run independently
6. **Eigen alignment** - respect `EIGEN_MAKE_ALIGNED_OPERATOR_NEW` macro for aligned types

## License

- `radar_odom`: Apache-2.0
- `temporal_radar_mapping`: MIT
