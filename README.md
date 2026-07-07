# Dual-Arm Synchronized Trajectory Generation

Synchronized trajectory generation for dual Doosan M1013 robots with hierarchical collision detection, RRT-Connect warm-start planning, and B-spline optimization.

Based on: **"Generation of Synchronized Configuration Space Trajectories of Multi-Robot Systems"** by Ariyan Kabir

## 🎯 Features

- **Graph-based Optimal IK**: Dynamic programming for optimal inverse kinematics through waypoints
- **Cubic B-spline Trajectories**: Smooth trajectory generation with continuous derivatives
- **Hierarchical Collision Detection**: 5-level subdivision (1→2→4→8→16→32 segments)
- **RRT-Connect with Warm-Start**: Uses failed B-spline segments to seed RRT trees
- **Informed RRT***: Biased sampling, cost optimization, and rewiring
- **B-spline Smoothing**: Reduces jerks and acceleration discontinuities
- **Full 6-DOF Support**: Complete forward kinematics for Doosan M1013

## 📋 Pipeline Overview

```
Waypoints → IK Solver → B-spline → Collision Detection → RRT (if needed) → Smoothing → Execution
     ↓           ↓           ↓              ↓                    ↓              ↓           ↓
Cartesian   Joint      Smooth      Hierarchical         Warm-started      Cubic       Robot
  [x,y,z]   Configs   Trajectory   1→2→4→8→16→32        RRT-Connect      B-spline    Commands
```

## 🚀 Installation

### Prerequisites

1. **ROS2 Humble** (or compatible version)
2. **Python 3.8+**
3. **Doosan Robot Packages** (from GitHub)

### Step 1: Install Dependencies

```bash
# ROS2 dependencies
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-ros-gz

# Python dependencies
pip install numpy scipy
```

### Step 2: Clone Doosan Robot Packages

```bash
cd ~/dual_arm_ws/src
git clone https://github.com/doosan-robotics/doosan-robot2.git
```

### Step 3: Clone This Package

```bash
cd ~/dual_arm_ws/src
# Copy the dual_arm_sync folder here
```

### Step 4: Build Workspace

```bash
cd ~/dual_arm_ws
colcon build --symlink-install
source install/setup.bash
```

## 🎮 Usage

### 1. Launch Gazebo with Dual Robots

```bash
ros2 launch dual_arm_sync dual_gazebo.launch.py
```

This will spawn two Doosan M1013 robots:
- `dsr01` at y=+0.5m (white)
- `dsr02` at y=-0.5m (blue)

### 2. Run the Complete Pipeline

```bash
ros2 launch dual_arm_sync dual_arm_pipeline.launch.py
```

Or run individual components:

```bash
# Test collision checker
ros2 run dual_arm_sync collision_checker

# Test IK solver
ros2 run dual_arm_sync optimal_ik_solver

# Test hierarchical collision detection
ros2 run dual_arm_sync hierarchical_collision

# Test RRT planner
ros2 run dual_arm_sync rrt_warmstart_planner

# Run main pipeline
ros2 run dual_arm_sync main_pipeline
```

### 3. Run with Custom Parameters

```bash
ros2 launch dual_arm_sync dual_arm_pipeline.launch.py \
  collision_threshold:=0.2 \
  rrt_max_iterations:=10000 \
  trajectory_duration:=15.0
```

## 📁 File Structure

```
dual_arm_ws/
├── src/
│   ├── dual_arm_sync/
│   │   ├── dual_arm_sync/
│   │   │   ├── __init__.py
│   │   │   ├── constants.py                  # Global constants
│   │   │   ├── collision_checker.py          # Fast collision detection
│   │   │   ├── optimal_ik_solver.py          # Graph-optimal IK with DP
│   │   │   ├── bspline_trajectory.py         # Cubic B-spline generation
│   │   │   ├── hierarchical_collision.py     # 5-level segmentation
│   │   │   ├── rrt_warmstart_planner.py      # RRT with warm-start
│   │   │   ├── trajectory_smoother.py        # B-spline smoothing
│   │   │   ├── trajectory_executor.py        # Execution & metrics
│   │   │   └── main_pipeline.py              # Main orchestrator
│   │   ├── launch/
│   │   │   ├── dual_gazebo.launch.py
│   │   │   └── dual_arm_pipeline.launch.py
│   │   ├── config/
│   │   │   └── pipeline_params.yaml
│   │   ├── rviz/
│   │   │   └── dual_arm.rviz
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── setup.cfg
```

## ⚙️ Configuration

Edit `config/pipeline_params.yaml` to customize:

```yaml
dual_arm_pipeline:
  ros__parameters:
    # Collision detection
    collision_threshold: 0.15
    hierarchical_max_depth: 5
    
    # RRT planning
    rrt_max_iterations: 5000
    rrt_step_size: 0.1
    use_informed_sampling: true
    
    # Trajectory
    trajectory_duration: 10.0
    control_rate: 100.0
```

## 🧪 Testing Individual Components

### Test Collision Checker (6-DOF)
```bash
ros2 run dual_arm_sync collision_checker
```

### Test IK Solver
```bash
ros2 run dual_arm_sync optimal_ik_solver
```

### Test Hierarchical Collision Detection
```bash
ros2 run dual_arm_sync hierarchical_collision
```

### Test RRT Planner
```bash
ros2 run dual_arm_sync rrt_warmstart_planner
```

### Test Trajectory Smoother
```bash
ros2 run dual_arm_sync trajectory_smoother
```

## 📊 Pipeline Stages

### 1. Optimal IK Solving
- Uses graph-based dynamic programming
- Finds optimal joint path through Cartesian waypoints
- Considers all IK solutions (typically 2-8 per waypoint)

### 2. B-spline Generation
- Creates cubic B-spline trajectories
- Ensures C² continuity (smooth acceleration)
- Generates synchronized trajectories for both robots

### 3. Hierarchical Collision Detection
- Starts with entire trajectory as single segment
- Recursively subdivides collision segments
- Depth 5: 1→2→4→8→16→32 segments
- Only subdivides segments with collisions

### 4. RRT-Connect with Warm-Start
- **Warm-Start**: Seeds RRT trees with failed B-spline configs
- **Informed Sampling**: Samples within ellipsoid after first solution
- **Biased Cost**: `cost = deviation_weight × dev + length_weight × len`
- **Rewiring**: RRT* style optimization for lower cost paths

### 5. B-spline Smoothing
- Smooths rough RRT paths
- Reduces jerks and acceleration spikes
- Maintains collision-free guarantee

### 6. Execution
- Publishes joint commands at 100 Hz
- Monitors tracking errors
- Computes execution metrics

## 🔧 Troubleshooting

### Issue: "No module named 'dual_arm_sync'"
**Solution**: Make sure you've built the workspace and sourced it:
```bash
cd ~/dual_arm_ws
colcon build --symlink-install
source install/setup.bash
```

### Issue: "Failed to find package 'dsr_gazebo2'"
**Solution**: Clone Doosan robot packages:
```bash
cd ~/dual_arm_ws/src
git clone https://github.com/doosan-robotics/doosan-robot2.git
colcon build
```

### Issue: Gazebo spawns two worlds
**Solution**: The launch file has been fixed to use `gz_sim.launch.py` directly instead of `dsr_gazebo.launch.py` to avoid duplicate spawning.

### Issue: RRT planning takes too long
**Solution**: 
- Increase `rrt_max_iterations` in config
- Reduce `hierarchical_max_depth` to get larger segments
- Increase `rrt_step_size` for faster exploration

### Issue: Collisions still detected after RRT
**Solution**:
- Decrease `collision_threshold` (but not too small)
- Increase `samples_per_segment` for more thorough checking
- Check if robots are too close at start/goal

## 📈 Performance Metrics

Typical performance on standard hardware:
- IK Solving: 0.1-0.5s
- B-spline Generation: <0.1s
- Collision Detection: 0.2-1.0s
- RRT Planning (per segment): 1-10s
- Total Pipeline: 2-15s (depending on collisions)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License

## 📚 References

1. Kabir, A. "Generation of Synchronized Configuration Space Trajectories of Multi-Robot Systems"
2. Doosan Robotics M1013 Technical Documentation
3. LaValle, S. M. "Planning Algorithms" (for RRT)
4. Kalman, R. "B-spline Fundamentals"

## 👤 Author

Your Name - your.email@example.com

## 🙏 Acknowledgments

- Ariyan Kabir for the hierarchical collision detection approach
- Doosan Robotics for M1013 robot specifications
- ROS2 community for excellent tools and documentation