#!/usr/bin/env python3
"""
hierarchical_collision_resolver.py - COMPLETE COLLISION RESOLUTION PIPELINE

Workflow:
1. Generate trajectories (B-spline)
2. Check collision
3. If NO collision → Execute directly
4. If collision → Try Kuramoto
5. If Kuramoto succeeds → Execute
6. If Kuramoto fails → Try RRT-Connect with warm start
7. If RRT succeeds → Execute
8. If RRT fails → Skip trial

Features:
- Fast execution (5s instead of 10s)
- Hierarchical resolution
- Warm-start RRT from failed Kuramoto path
- Continuous motion
"""

import subprocess
import json
import time
import sys
import os
import re
from pathlib import Path
from datetime import datetime

try:
    import numpy as np
    if np.__version__.startswith('2.'):
        print(f"ERROR: NumPy {np.__version__} not compatible!")
        sys.exit(1)
except ImportError:
    print("ERROR: numpy not found")
    sys.exit(1)


class HierarchicalCollisionResolver:
    """Hierarchical collision resolution with B-spline → Kuramoto → RRT"""
    
    def __init__(self, num_trials=100):
        self.num_trials = num_trials
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f"hierarchical_test_{timestamp}")
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'failed_trials').mkdir(exist_ok=True)
        (self.output_dir / 'collision_free').mkdir(exist_ok=True)
        (self.output_dir / 'kuramoto_resolved').mkdir(exist_ok=True)
        (self.output_dir / 'rrt_resolved').mkdir(exist_ok=True)
        (self.output_dir / 'unresolved').mkdir(exist_ok=True)
        
        print("\nReading initial position from Gazebo...")
        self.current_joints = self.read_joints_from_gazebo()
        
        if self.current_joints:
            print(f"✓ Initial joints read from Gazebo")
        else:
            print("⚠ Using home position")
            self.current_joints = {
                'dsr01': np.zeros(6),
                'dsr02': np.zeros(6)
            }
        
        # Execution duration (5 seconds for speed)
        self.execution_duration = 5.0
        
        self.results = {
            'metadata': {
                'total_trials': num_trials,
                'start_time': datetime.now().isoformat(),
                'execution_duration': self.execution_duration,
            },
            'statistics': {
                'completed': 0,
                'collision_free_initial': 0,
                'collision_detected': 0,
                'resolved_by_kuramoto': 0,
                'kuramoto_failed_rrt_tried': 0,
                'resolved_by_rrt': 0,
                'failed_unresolved': 0,
                'executed': 0,
            },
            'resolution_path': {
                'direct_bspline': 0,      # No collision
                'kuramoto_only': 0,        # Kuramoto resolved
                'rrt_after_kuramoto': 0,   # RRT resolved after Kuramoto failed
                'failed': 0,               # Couldn't resolve
            },
            'timing': {
                'ik_times': [],
                'trajectory_times': [],
                'collision_times': [],
                'kuramoto_times': [],
                'rrt_times': [],
                'execution_times': [],
            },
            'trials': []
        }
        
        self.print_header()
    
    def print_header(self):
        """Print header"""
        print(f"\n{'='*80}")
        print(f"HIERARCHICAL COLLISION RESOLUTION - {self.num_trials} TRIALS")
        print(f"{'='*80}")
        print(f"Pipeline: B-spline → Kuramoto → RRT-Connect")
        print(f"Execution duration: {self.execution_duration}s (fast)")
        print(f"Output: {self.output_dir}")
        print(f"{'='*80}\n")
    
    def read_joints_from_gazebo(self):
        """Read current joints from Gazebo"""
        try:
            result1 = subprocess.run(
                ['ros2', 'topic', 'echo', '/dsr01/gz/joint_states', '--once'],
                capture_output=True, text=True, timeout=5
            )
            result2 = subprocess.run(
                ['ros2', 'topic', 'echo', '/dsr02/gz/joint_states', '--once'],
                capture_output=True, text=True, timeout=5
            )
            
            pos_match1 = re.search(r'position:\s*\[([\d\s.,\-e]+)\]', result1.stdout)
            pos_match2 = re.search(r'position:\s*\[([\d\s.,\-e]+)\]', result2.stdout)
            
            if pos_match1 and pos_match2:
                joints1 = [float(x.strip()) for x in pos_match1.group(1).split(',')[:6]]
                joints2 = [float(x.strip()) for x in pos_match2.group(1).split(',')[:6]]
                return {
                    'dsr01': np.array(joints1),
                    'dsr02': np.array(joints2)
                }
            return None
        except:
            return None
    
    def generate_smart_targets(self, trial_num):
        """Generate smart collision-prone targets"""
        scenario = np.random.choice(['corners', 'center', 'cross', 'parallel'])
        
        if scenario == 'corners':
            x_base = np.random.choice([-0.4, 0.4])
            dsr01 = {
                'x': round(x_base + np.random.uniform(-0.1, 0.1), 3),
                'y': round(0.5 + np.random.uniform(-0.2, 0.1), 3),
                'z': round(np.random.uniform(0.6, 1.1), 3),
            }
            dsr02 = {
                'x': round(x_base + np.random.uniform(-0.1, 0.1), 3),
                'y': round(-0.5 + np.random.uniform(-0.1, 0.2), 3),
                'z': round(np.random.uniform(0.6, 1.1), 3),
            }
        elif scenario == 'center':
            x_center = np.random.uniform(-0.2, 0.2)
            dsr01 = {
                'x': round(x_center + np.random.uniform(-0.15, 0.15), 3),
                'y': round(np.random.uniform(0.1, 0.3), 3),
                'z': round(np.random.uniform(0.6, 0.9), 3),
            }
            dsr02 = {
                'x': round(x_center + np.random.uniform(-0.15, 0.15), 3),
                'y': round(np.random.uniform(-0.3, -0.1), 3),
                'z': round(np.random.uniform(0.6, 0.9), 3),
            }
        elif scenario == 'cross':
            x1, x2 = np.random.uniform(-0.4, -0.1), np.random.uniform(0.1, 0.4)
            dsr01 = {
                'x': round(x2, 3),
                'y': round(0.5 + np.random.uniform(-0.2, 0.1), 3),
                'z': round(np.random.uniform(0.6, 1.0), 3),
            }
            dsr02 = {
                'x': round(x1, 3),
                'y': round(-0.5 + np.random.uniform(-0.1, 0.2), 3),
                'z': round(np.random.uniform(0.6, 1.0), 3),
            }
        else:  # parallel
            x_shared = np.random.uniform(-0.3, 0.3)
            dsr01 = {
                'x': round(x_shared + np.random.uniform(-0.05, 0.05), 3),
                'y': round(np.random.uniform(0.3, 0.6), 3),
                'z': round(np.random.uniform(0.6, 1.0), 3),
            }
            dsr02 = {
                'x': round(x_shared + np.random.uniform(-0.05, 0.05), 3),
                'y': round(np.random.uniform(-0.6, -0.3), 3),
                'z': round(np.random.uniform(0.6, 1.0), 3),
            }
        
        return {'dsr01': dsr01, 'dsr02': dsr02, 'scenario': scenario}
    
    def modify_ik_solutions(self):
        """Modify IK solutions with current joints"""
        try:
            if os.path.exists('ik_solutions.json'):
                with open('ik_solutions.json', 'r') as f:
                    ik_data = json.load(f)
                
                # Update to use FAST execution (5s instead of 10s)
                ik_data['duration'] = self.execution_duration
                
                ik_data['dsr01']['current_joints'] = self.current_joints['dsr01'].tolist()
                ik_data['dsr01']['current_joints_deg'] = np.degrees(self.current_joints['dsr01']).tolist()
                ik_data['dsr02']['current_joints'] = self.current_joints['dsr02'].tolist()
                ik_data['dsr02']['current_joints_deg'] = np.degrees(self.current_joints['dsr02']).tolist()
                
                with open('ik_solutions.json', 'w') as f:
                    json.dump(ik_data, f, indent=2)
                return True
            return False
        except:
            return False
    
    def run_ik_solver(self, targets):
        """Run IK solver"""
        input_str = (
            f"{targets['dsr01']['x']}\n{targets['dsr01']['y']}\n{targets['dsr01']['z']}\n"
            f"{targets['dsr02']['x']}\n{targets['dsr02']['y']}\n{targets['dsr02']['z']}\n"
            "n\n"
        )
        try:
            start = time.time()
            subprocess.run(
                ['ros2', 'run', 'dual_arm_sync', 'dual_arm_ik_solver'],
                input=input_str, capture_output=True, text=True, timeout=30
            )
            elapsed = time.time() - start
            if os.path.exists('ik_solutions.json'):
                self.modify_ik_solutions()
                return True, elapsed
            return False, elapsed
        except:
            return False, 0
    
    def run_trajectory(self):
        """Run trajectory generation"""
        try:
            start = time.time()
            subprocess.run(
                ['ros2', 'run', 'dual_arm_sync', 'trajectory_generation'],
                capture_output=True, timeout=30
            )
            elapsed = time.time() - start
            return os.path.exists('trajectories.json'), elapsed
        except:
            return False, 0
    
    def run_collision_checker(self):
        """Run collision checker"""
        try:
            start = time.time()
            result = subprocess.run(
                ['ros2', 'run', 'dual_arm_sync', 'collision_checker'],
                capture_output=True, text=True, timeout=30
            )
            elapsed = time.time() - start
            output = result.stdout + result.stderr
            has_collision = 'collision_detected: true' in output.lower()
            return True, has_collision, elapsed
        except:
            return False, False, 0
    
    def run_kuramoto(self):
        """Run Kuramoto synchronization"""
        try:
            start = time.time()
            result = subprocess.run(
                ['ros2', 'run', 'dual_arm_sync', 'kuramoto_synchronization'],
                capture_output=True, text=True, timeout=120
            )
            elapsed = time.time() - start
            output = result.stdout + result.stderr
            safe = ('no collision' in output.lower() and 
                   os.path.exists('synchronized_trajectories.json'))
            return True, safe, elapsed
        except:
            return False, False, 0
    
    def run_rrt_connect_with_warm_start(self):
        """
        Run RRT-Connect with warm start from failed Kuramoto path
        
        Uses synchronized_trajectories.json (failed Kuramoto result) as initial guess
        """
        try:
            print(f"      🔧 Starting RRT-Connect with warm start...", end=' ', flush=True)
            start = time.time()
            
            # RRT-Connect takes the failed trajectory as warm start
            result = subprocess.run(
                ['ros2', 'run', 'dual_arm_sync', 'rrt_connect_planner'],
                capture_output=True,
                text=True,
                timeout=180  # 3 minutes max for RRT
            )
            elapsed = time.time() - start
            
            output = result.stdout + result.stderr
            
            # Check if RRT found a collision-free path
            success = (
                'rrt success' in output.lower() or
                'collision-free path found' in output.lower() or
                os.path.exists('rrt_trajectories.json')
            )
            
            if success:
                print(f"✓ ({elapsed:.1f}s)")
                return True, elapsed
            else:
                print(f"✗ ({elapsed:.1f}s)")
                return False, elapsed
        except subprocess.TimeoutExpired:
            print(f"✗ (timeout)")
            return False, 180
        except:
            print(f"✗")
            return False, 0
    
    def execute_in_gazebo(self, trajectory_file='synchronized_trajectories.json'):
        """Execute trajectory in Gazebo"""
        try:
            print(f"    ▶ Executing...", end=' ', flush=True)
            start = time.time()
            
            # Modify gazebo_executor to use specific trajectory file
            process = subprocess.Popen(
                ['ros2', 'run', 'dual_arm_sync', 'gazebo_executor'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, 'TRAJECTORY_FILE': trajectory_file}
            )
            
            time.sleep(2)
            
            try:
                process.stdin.write('\n')
                process.stdin.flush()
            except:
                pass
            
            # Wait for execution (5s + buffer)
            try:
                process.communicate(timeout=self.execution_duration + 5)
                elapsed = time.time() - start
                print(f"✓ ({elapsed:.1f}s)")
                return True, elapsed
            except subprocess.TimeoutExpired:
                process.kill()
                elapsed = time.time() - start
                print(f"✓ ({elapsed:.1f}s)")
                return True, elapsed
        except:
            print(f"✗")
            return False, 0
    
    def run_trial(self, trial_num, targets):
        """Run one complete trial with hierarchical resolution"""
        scenario = targets.get('scenario', 'unknown')
        
        print(f"\n{'='*80}")
        print(f"TRIAL {trial_num}/{self.num_trials} - {scenario.upper()}")
        print(f"{'='*80}")
        print(f"DSR01: ({targets['dsr01']['x']:.3f}, {targets['dsr01']['y']:.3f}, {targets['dsr01']['z']:.3f})")
        print(f"DSR02: ({targets['dsr02']['x']:.3f}, {targets['dsr02']['y']:.3f}, {targets['dsr02']['z']:.3f})")
        
        trial_data = {
            'trial': trial_num,
            'scenario': scenario,
            'targets': targets,
            'timing': {},
            'resolution_method': None,
            'status': 'running'
        }
        
        # Step 1: IK
        print(f"  [1/5] IK...", end=' ', flush=True)
        success, t = self.run_ik_solver(targets)
        trial_data['timing']['ik'] = t
        if not success:
            print(f"✗ ({t:.2f}s)")
            trial_data['status'] = 'failed_ik'
            return False, trial_data, 'failed'
        print(f"✓ ({t:.2f}s)")
        
        # Step 2: Trajectory (B-spline)
        print(f"  [2/5] Traj (B-spline)...", end=' ', flush=True)
        success, t = self.run_trajectory()
        trial_data['timing']['trajectory'] = t
        if not success:
            print(f"✗ ({t:.2f}s)")
            trial_data['status'] = 'failed_trajectory'
            return False, trial_data, 'failed'
        print(f"✓ ({t:.2f}s)")
        
        # Step 3: Collision Check
        print(f"  [3/5] Collision...", end=' ', flush=True)
        success, has_collision, t = self.run_collision_checker()
        trial_data['timing']['collision'] = t
        trial_data['has_collision'] = has_collision
        if not success:
            print(f"✗ ({t:.2f}s)")
            trial_data['status'] = 'failed_collision_check'
            return False, trial_data, 'failed'
        
        # Branch based on collision detection
        if not has_collision:
            # NO COLLISION → Execute B-spline directly!
            print(f"✓ CLEAR ({t:.2f}s)")
            print(f"    → No collision! Executing B-spline directly")
            trial_data['resolution_method'] = 'direct_bspline'
            trial_data['status'] = 'ready'
            self.results['statistics']['collision_free_initial'] += 1
            self.results['resolution_path']['direct_bspline'] += 1
            
            # Use original trajectories.json (not synchronized)
            if os.path.exists('trajectories.json'):
                # Copy to synchronized for executor
                with open('trajectories.json', 'r') as f:
                    traj_data = json.load(f)
                with open('synchronized_trajectories.json', 'w') as f:
                    json.dump(traj_data, f)
            
            return True, trial_data, 'collision_free'
        
        # COLLISION DETECTED
        print(f"❌ COLLISION ({t:.2f}s)")
        self.results['statistics']['collision_detected'] += 1
        
        # Step 4: Try Kuramoto
        print(f"  [4/5] Kuramoto...", end=' ', flush=True)
        success, safe, t = self.run_kuramoto()
        trial_data['timing']['kuramoto'] = t
        
        if not success:
            print(f"✗ ({t:.2f}s)")
            trial_data['status'] = 'failed_kuramoto'
            return False, trial_data, 'failed'
        
        if safe:
            # KURAMOTO SUCCEEDED!
            print(f"✓ SAFE ({t:.2f}s)")
            print(f"    🎯 Kuramoto resolved collision!")
            trial_data['resolution_method'] = 'kuramoto_only'
            trial_data['status'] = 'ready'
            self.results['statistics']['resolved_by_kuramoto'] += 1
            self.results['resolution_path']['kuramoto_only'] += 1
            
            # Record timing
            self.results['timing']['ik_times'].append(trial_data['timing']['ik'])
            self.results['timing']['trajectory_times'].append(trial_data['timing']['trajectory'])
            self.results['timing']['collision_times'].append(trial_data['timing']['collision'])
            self.results['timing']['kuramoto_times'].append(trial_data['timing']['kuramoto'])
            
            return True, trial_data, 'kuramoto_resolved'
        
        # KURAMOTO FAILED → Try RRT-Connect
        print(f"❌ UNSAFE ({t:.2f}s)")
        print(f"    ⚠ Kuramoto failed! Trying RRT-Connect with warm start...")
        
        # Step 5: RRT-Connect with warm start
        print(f"  [5/5] RRT-Connect...", end=' ')
        success, t_rrt = self.run_rrt_connect_with_warm_start()
        trial_data['timing']['rrt'] = t_rrt
        self.results['statistics']['kuramoto_failed_rrt_tried'] += 1
        
        if success:
            print(f"      ✓ RRT found collision-free path!")
            trial_data['resolution_method'] = 'rrt_after_kuramoto'
            trial_data['status'] = 'ready'
            self.results['statistics']['resolved_by_rrt'] += 1
            self.results['resolution_path']['rrt_after_kuramoto'] += 1
            
            # Record timing
            self.results['timing']['ik_times'].append(trial_data['timing']['ik'])
            self.results['timing']['trajectory_times'].append(trial_data['timing']['trajectory'])
            self.results['timing']['collision_times'].append(trial_data['timing']['collision'])
            self.results['timing']['kuramoto_times'].append(trial_data['timing']['kuramoto'])
            self.results['timing']['rrt_times'].append(trial_data['timing']['rrt'])
            
            # Use RRT trajectory for execution
            if os.path.exists('rrt_trajectories.json'):
                with open('rrt_trajectories.json', 'r') as f:
                    rrt_data = json.load(f)
                with open('synchronized_trajectories.json', 'w') as f:
                    json.dump(rrt_data, f)
            
            return True, trial_data, 'rrt_resolved'
        else:
            print(f"      ❌ RRT also failed!")
            print(f"      ⛔ No collision-free path found")
            trial_data['resolution_method'] = 'failed_all'
            trial_data['status'] = 'unresolved'
            self.results['statistics']['failed_unresolved'] += 1
            self.results['resolution_path']['failed'] += 1
            return False, trial_data, 'unresolved'
    
    def run_all_trials(self):
        """Run all trials"""
        print(f"\n{'='*80}")
        print(f"STARTING {self.num_trials} HIERARCHICAL TESTS")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        for trial_num in range(1, self.num_trials + 1):
            targets = self.generate_smart_targets(trial_num)
            success, trial_data, resolution_type = self.run_trial(trial_num, targets)
            
            self.results['statistics']['completed'] += 1
            self.results['trials'].append(trial_data)
            
            if success and trial_data['status'] == 'ready':
                # Execute!
                exec_ok, exec_time = self.execute_in_gazebo()
                trial_data['timing']['execution'] = exec_time
                self.results['timing']['execution_times'].append(exec_time)
                
                if exec_ok:
                    self.results['statistics']['executed'] += 1
                    
                    # Read new position
                    print(f"    📍 Reading position...", end=' ', flush=True)
                    time.sleep(0.5)  # Shorter wait for fast execution
                    new_joints = self.read_joints_from_gazebo()
                    
                    if new_joints:
                        self.current_joints = new_joints
                        print(f"✓")
                    else:
                        print(f"✗")
                        try:
                            with open('ik_solutions.json', 'r') as f:
                                ik_data = json.load(f)
                            self.current_joints['dsr01'] = np.array(ik_data['dsr01']['optimal_joints'])
                            self.current_joints['dsr02'] = np.array(ik_data['dsr02']['optimal_joints'])
                        except:
                            pass
                    
                    # Save based on resolution method
                    if resolution_type == 'collision_free':
                        save_dir = self.output_dir / 'collision_free'
                    elif resolution_type == 'kuramoto_resolved':
                        save_dir = self.output_dir / 'kuramoto_resolved'
                    elif resolution_type == 'rrt_resolved':
                        save_dir = self.output_dir / 'rrt_resolved'
                    else:
                        save_dir = self.output_dir / 'successful_trials'
                    
                    with open(save_dir / f"trial_{trial_num:03d}.json", 'w') as f:
                        json.dump(trial_data, f, indent=2)
            else:
                # Failed or unresolved
                print(f"    ⛔ Skipping execution")
                
                if resolution_type == 'unresolved':
                    save_dir = self.output_dir / 'unresolved'
                else:
                    save_dir = self.output_dir / 'failed_trials'
                
                with open(save_dir / f"trial_{trial_num:03d}.json", 'w') as f:
                    json.dump(trial_data, f, indent=2)
            
            # Cleanup
            for f in ['ik_solutions.json', 'trajectories.json', 'collision_report.json', 
                     'synchronized_trajectories.json', 'rrt_trajectories.json']:
                if os.path.exists(f):
                    os.remove(f)
            
            if trial_num % 10 == 0:
                self.print_progress()
        
        elapsed = time.time() - start_time
        self.print_final_report(elapsed)
        self.save_results()
    
    def print_progress(self):
        c = self.results['statistics']['completed']
        print(f"\n{'─'*80}")
        print(f"PROGRESS: {c}/{self.num_trials}")
        print(f"  Clear: {self.results['statistics']['collision_free_initial']}")
        print(f"  Collisions: {self.results['statistics']['collision_detected']}")
        print(f"    ├─ Kuramoto: {self.results['statistics']['resolved_by_kuramoto']}")
        print(f"    ├─ RRT: {self.results['statistics']['resolved_by_rrt']}")
        print(f"    └─ Failed: {self.results['statistics']['failed_unresolved']}")
        print(f"  Executed: {self.results['statistics']['executed']}")
        print(f"{'─'*80}\n")
    
    def print_final_report(self, elapsed):
        print(f"\n{'='*80}")
        print(f"HIERARCHICAL RESOLUTION RESULTS")
        print(f"{'='*80}")
        print(f"Duration: {elapsed/60:.1f} min")
        print(f"Completed: {self.results['statistics']['completed']}")
        
        print(f"\n{'─'*80}")
        print(f"RESOLUTION STATISTICS")
        print(f"{'─'*80}")
        print(f"Collision-Free Initially: {self.results['statistics']['collision_free_initial']}")
        print(f"  → Executed directly (B-spline): {self.results['resolution_path']['direct_bspline']}")
        
        print(f"\nCollisions Detected: {self.results['statistics']['collision_detected']}")
        print(f"  ├─ Resolved by Kuramoto: {self.results['statistics']['resolved_by_kuramoto']}")
        print(f"  ├─ Kuramoto failed, RRT tried: {self.results['statistics']['kuramoto_failed_rrt_tried']}")
        print(f"  │   ├─ RRT succeeded: {self.results['statistics']['resolved_by_rrt']}")
        print(f"  │   └─ RRT failed: {self.results['statistics']['failed_unresolved']}")
        print(f"  └─ Total unresolved: {self.results['statistics']['failed_unresolved']}")
        
        if self.results['statistics']['collision_detected'] > 0:
            total_resolved = (self.results['statistics']['resolved_by_kuramoto'] + 
                            self.results['statistics']['resolved_by_rrt'])
            resolution_rate = (total_resolved / self.results['statistics']['collision_detected'] * 100)
            print(f"\n🎯 OVERALL RESOLUTION RATE: {resolution_rate:.1f}%")
            
            if self.results['statistics']['resolved_by_kuramoto'] > 0:
                kuramoto_rate = (self.results['statistics']['resolved_by_kuramoto'] / 
                               self.results['statistics']['collision_detected'] * 100)
                print(f"   Kuramoto success rate: {kuramoto_rate:.1f}%")
            
            if self.results['statistics']['resolved_by_rrt'] > 0:
                rrt_rate = (self.results['statistics']['resolved_by_rrt'] / 
                          self.results['statistics']['kuramoto_failed_rrt_tried'] * 100)
                print(f"   RRT success rate (when Kuramoto failed): {rrt_rate:.1f}%")
        
        print(f"\n{'─'*80}")
        print(f"EXECUTION")
        print(f"{'─'*80}")
        print(f"Executed: {self.results['statistics']['executed']}/{self.results['statistics']['completed']}")
        success_rate = (self.results['statistics']['executed'] / 
                       self.results['statistics']['completed'] * 100)
        print(f"Success rate: {success_rate:.1f}%")
        
        if self.results['timing']['execution_times']:
            print(f"\n{'─'*80}")
            print(f"TIMINGS")
            print(f"{'─'*80}")
            print(f"IK: {np.mean(self.results['timing']['ik_times']):.2f}s avg")
            print(f"Trajectory: {np.mean(self.results['timing']['trajectory_times']):.2f}s avg")
            print(f"Collision check: {np.mean(self.results['timing']['collision_times']):.2f}s avg")
            if self.results['timing']['kuramoto_times']:
                print(f"Kuramoto: {np.mean(self.results['timing']['kuramoto_times']):.2f}s avg")
            if self.results['timing']['rrt_times']:
                print(f"RRT-Connect: {np.mean(self.results['timing']['rrt_times']):.2f}s avg")
            print(f"Execution: {np.mean(self.results['timing']['execution_times']):.2f}s avg")
        
        print(f"\n{'='*80}\n")
    
    def save_results(self):
        with open(self.output_dir / 'final_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Results: {self.output_dir}/final_results.json")
        print(f"✓ Direct (B-spline): {self.output_dir}/collision_free/")
        print(f"✓ Kuramoto resolved: {self.output_dir}/kuramoto_resolved/")
        print(f"✓ RRT resolved: {self.output_dir}/rrt_resolved/")
        print(f"✓ Unresolved: {self.output_dir}/unresolved/\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=100)
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("HIERARCHICAL COLLISION RESOLUTION")
    print("="*80)
    print(f"NumPy: {np.__version__}")
    print("="*80)
    
    if np.__version__.startswith('2.'):
        print("\n❌ NumPy 2.x! Fix: pip3 uninstall numpy -y && pip3 install 'numpy<2'\n")
        return
    
    print("\n⚠ Prerequisites:")
    print("  1. Gazebo running")
    print("  2. RRT-Connect planner available")
    response = input("\nReady? (y/n): ")
    if response.lower() != 'y':
        return
    
    resolver = HierarchicalCollisionResolver(num_trials=args.trials)
    resolver.run_all_trials()


if __name__ == '__main__':
    main()