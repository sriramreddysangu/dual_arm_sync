#!/usr/bin/env python3
"""
Enhanced Automated Dual-Arm Testing Orchestrator
Includes position configuration files and better integration with existing nodes

Usage:
    python3 enhanced_testing_orchestrator.py --trials 100 --workspace-config workspace.yaml
"""

import subprocess
import json
import yaml
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse
import sys
import os
import shutil


class PositionGenerator:
    """Generates valid cartesian positions for dual-arm trials"""
    
    def __init__(self, config_file: Optional[str] = None):
        # CORRECTED: Default workspace bounds from constants.py
        # Robot home position has end-effector at z = 1.4525m
        # Maximum reach is 1.3m, workspace extends to z = 1.5m
        self.config = {
            'workspace_bounds': {
                'x_min': -0.8, 'x_max': 0.8,
                'y_min': -1.0, 'y_max': 1.0,
                'z_min': 0.0, 'z_max': 1.5  # CORRECTED from 1.0 to 1.5
            },
            'robot1_base_y': 0.5,
            'robot2_base_y': -0.5,
            'min_arm_separation': 0.3,
            'min_movement_distance': 0.1,
            'max_movement_distance': 0.6,
            # Additional constraints for realistic working heights
            'working_z_min': 0.2,  # Stay above ground
            'working_z_max': 1.4   # Stay within comfortable reach
        }
        
        # Load custom config if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                custom_config = yaml.safe_load(f)
                self.config.update(custom_config)
    
    def generate_position_set(self, strategy: str = 'random') -> Tuple[Dict, Dict]:
        """
        Generate a set of positions for both arms
        
        Args:
            strategy: 'random', 'systematic', 'challenging'
        
        Returns:
            Tuple of (arm1_positions, arm2_positions)
        """
        if strategy == 'random':
            return self._generate_random_positions()
        elif strategy == 'systematic':
            return self._generate_systematic_positions()
        elif strategy == 'challenging':
            return self._generate_challenging_positions()
        else:
            return self._generate_random_positions()
    
    def _generate_random_positions(self) -> Tuple[Dict, Dict]:
        """Generate random positions with safety constraints"""
        wb = self.config['workspace_bounds']
        # Use working Z range for more realistic positions
        z_min = self.config.get('working_z_min', wb['z_min'])
        z_max = self.config.get('working_z_max', wb['z_max'])
        
        max_attempts = 100
        for attempt in range(max_attempts):
            # Arm1 (right side, y > 0)
            arm1_start = {
                'x': np.random.uniform(wb['x_min'], wb['x_max']),
                'y': np.random.uniform(0.2, wb['y_max']),
                'z': np.random.uniform(z_min, z_max)
            }
            
            # Generate target with constrained movement
            movement_dist = np.random.uniform(
                self.config['min_movement_distance'],
                self.config['max_movement_distance']
            )
            
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            
            arm1_target = {
                'x': arm1_start['x'] + direction[0] * movement_dist,
                'y': arm1_start['y'] + direction[1] * movement_dist,
                'z': arm1_start['z'] + direction[2] * movement_dist
            }
            
            # Clamp to workspace
            arm1_target = {
                'x': np.clip(arm1_target['x'], wb['x_min'], wb['x_max']),
                'y': np.clip(arm1_target['y'], 0.2, wb['y_max']),
                'z': np.clip(arm1_target['z'], z_min, z_max)
            }
            
            # Arm2 (left side, y < 0)
            arm2_start = {
                'x': np.random.uniform(wb['x_min'], wb['x_max']),
                'y': np.random.uniform(wb['y_min'], -0.2),
                'z': np.random.uniform(z_min, z_max)
            }
            
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            
            arm2_target = {
                'x': arm2_start['x'] + direction[0] * movement_dist,
                'y': arm2_start['y'] + direction[1] * movement_dist,
                'z': arm2_start['z'] + direction[2] * movement_dist
            }
            
            arm2_target = {
                'x': np.clip(arm2_target['x'], wb['x_min'], wb['x_max']),
                'y': np.clip(arm2_target['y'], wb['y_min'], -0.2),
                'z': np.clip(arm2_target['z'], z_min, z_max)
            }
            
            # Validate separation
            if self._validate_positions(arm1_start, arm1_target, arm2_start, arm2_target):
                return (
                    {'start': arm1_start, 'target': arm1_target},
                    {'start': arm2_start, 'target': arm2_target}
                )
        
        # Fallback to safe positions
        print("⚠ Using fallback safe positions")
        return self._get_safe_default_positions()
    
    def _generate_challenging_positions(self) -> Tuple[Dict, Dict]:
        """Generate positions that are likely to cause collisions (for testing)"""
        wb = self.config['workspace_bounds']
        
        # Create crossing paths
        arm1_start = {
            'x': 0.6,
            'y': 0.5,
            'z': 0.5
        }
        
        arm1_target = {
            'x': -0.6,
            'y': 0.5,
            'z': 0.5
        }
        
        arm2_start = {
            'x': -0.6,
            'y': -0.5,
            'z': 0.5
        }
        
        arm2_target = {
            'x': 0.6,
            'y': -0.5,
            'z': 0.5
        }
        
        # Add some randomness
        noise = 0.1
        for pos in [arm1_start, arm1_target, arm2_start, arm2_target]:
            pos['x'] += np.random.uniform(-noise, noise)
            pos['z'] += np.random.uniform(-noise, noise)
        
        return (
            {'start': arm1_start, 'target': arm1_target},
            {'start': arm2_start, 'target': arm2_target}
        )
    
    def _generate_systematic_positions(self) -> Tuple[Dict, Dict]:
        """Generate positions in a systematic grid pattern"""
        # This would implement a systematic sweep of the workspace
        # For now, use random with higher density
        return self._generate_random_positions()
    
    def _validate_positions(self, a1_start, a1_target, a2_start, a2_target) -> bool:
        """Check if positions meet safety constraints"""
        def dist(p1, p2):
            return np.sqrt((p1['x']-p2['x'])**2 + (p1['y']-p2['y'])**2 + (p1['z']-p2['z'])**2)
        
        min_sep = self.config['min_arm_separation']
        
        # Check all combinations
        pairs = [
            (a1_start, a2_start),
            (a1_start, a2_target),
            (a1_target, a2_start),
            (a1_target, a2_target)
        ]
        
        for p1, p2 in pairs:
            if dist(p1, p2) < min_sep:
                return False
        
        return True
    
    def _get_safe_default_positions(self) -> Tuple[Dict, Dict]:
        """Return known safe default positions"""
        arm1 = {
            'start': {'x': 0.5, 'y': 0.5, 'z': 0.5},
            'target': {'x': 0.3, 'y': 0.6, 'z': 0.4}
        }
        
        arm2 = {
            'start': {'x': 0.5, 'y': -0.5, 'z': 0.5},
            'target': {'x': 0.3, 'y': -0.6, 'z': 0.4}
        }
        
        return arm1, arm2


class EnhancedTestOrchestrator:
    """Enhanced orchestrator with better ROS2 integration"""
    
    def __init__(self, num_trials: int = 100, output_dir: str = "test_results",
                 workspace_config: Optional[str] = None):
        self.num_trials = num_trials
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'trials').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'trajectories').mkdir(exist_ok=True)
        
        # Position generator
        self.position_gen = PositionGenerator(workspace_config)
        
        # Results tracking
        self.results = {
            'metadata': {
                'total_trials': num_trials,
                'start_time': datetime.now().isoformat(),
                'workspace_config': workspace_config
            },
            'statistics': {
                'completed_trials': 0,
                'failed_trials': 0,
                'collision_at_checker': 0,
                'collision_free_at_checker': 0,
                'resolved_by_kuramoto': 0,
                'failed_after_kuramoto': 0,
                'successful_executions': 0
            },
            'trials': []
        }
        
        print("="*80)
        print("ENHANCED DUAL-ARM AUTOMATED TESTING ORCHESTRATOR")
        print("="*80)
        print(f"Trials: {num_trials}")
        print(f"Output: {self.output_dir}")
        print(f"Config: {workspace_config or 'default'}")
        print("="*80 + "\n")
    
    def create_position_input_file(self, trial_num: int, arm1_pos: Dict, 
                                    arm2_pos: Dict, filename: str = "positions.yaml"):
        """Create position configuration file for the pipeline"""
        config = {
            'trial_number': trial_num,
            'timestamp': datetime.now().isoformat(),
            'dsr01': {
                'start': {
                    'x': float(arm1_pos['start']['x']),
                    'y': float(arm1_pos['start']['y']),
                    'z': float(arm1_pos['start']['z'])
                },
                'target': {
                    'x': float(arm1_pos['target']['x']),
                    'y': float(arm1_pos['target']['y']),
                    'z': float(arm1_pos['target']['z'])
                }
            },
            'dsr02': {
                'start': {
                    'x': float(arm2_pos['start']['x']),
                    'y': float(arm2_pos['start']['y']),
                    'z': float(arm2_pos['start']['z'])
                },
                'target': {
                    'x': float(arm2_pos['target']['x']),
                    'y': float(arm2_pos['target']['y']),
                    'z': float(arm2_pos['target']['z'])
                }
            }
        }
        
        with open(filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return filename
    
    def run_pipeline_step(self, step_name: str, command: List[str], 
                         timeout: int = 120) -> Tuple[bool, str, Dict]:
        """
        Run a single pipeline step
        
        Returns:
            Tuple of (success, output, parsed_data)
        """
        print(f"\n{'='*60}")
        print(f"Running: {step_name}")
        print(f"{'='*60}")
        
        log_file = self.output_dir / 'logs' / f"{step_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            
            output = result.stdout + result.stderr
            success = result.returncode == 0
            
            # Save log
            with open(log_file, 'w') as f:
                f.write(f"Command: {' '.join(command)}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"{'='*60}\n")
                f.write(output)
            
            # Parse output for key information
            parsed_data = self._parse_output(step_name, output)
            
            if success:
                print(f"✓ {step_name} completed successfully")
            else:
                print(f"❌ {step_name} failed (return code: {result.returncode})")
            
            return success, output, parsed_data
            
        except subprocess.TimeoutExpired:
            print(f"⚠ {step_name} timed out after {timeout}s")
            return False, f"Timeout after {timeout}s", {}
        except Exception as e:
            print(f"❌ Error running {step_name}: {e}")
            return False, str(e), {}
    
    def _parse_output(self, step_name: str, output: str) -> Dict:
        """Parse output from different pipeline steps"""
        parsed = {}
        
        if step_name == 'collision_checker':
            parsed['has_collision'] = (
                'COLLISION DETECTED' in output or 
                'collision detected' in output.lower()
            )
            # Try to extract collision count
            import re
            matches = re.search(r'(\d+)\s+collisions?', output.lower())
            if matches:
                parsed['collision_count'] = int(matches.group(1))
        
        elif step_name == 'kuramoto_synchronization':
            parsed['collision_free'] = (
                'collision-free' in output.lower() or
                'no collision' in output.lower() or
                os.path.exists('synchronized_trajectories.json')
            )
            # Try to extract optimization info
            if 'iterations' in output.lower():
                import re
                matches = re.search(r'(\d+)\s+iterations', output.lower())
                if matches:
                    parsed['iterations'] = int(matches.group(1))
        
        return parsed
    
    def run_single_trial(self, trial_num: int, strategy: str = 'random') -> Dict:
        """Run a complete trial"""
        print("\n" + "="*80)
        print(f"TRIAL {trial_num}/{self.num_trials}")
        print("="*80)
        
        trial_start_time = time.time()
        
        # Generate positions
        arm1_pos, arm2_pos = self.position_gen.generate_position_set(strategy)
        
        # Display positions
        print(f"\nPositions:")
        print(f"  Arm1: ({arm1_pos['start']['x']:.3f}, {arm1_pos['start']['y']:.3f}, {arm1_pos['start']['z']:.3f})")
        print(f"     → ({arm1_pos['target']['x']:.3f}, {arm1_pos['target']['y']:.3f}, {arm1_pos['target']['z']:.3f})")
        print(f"  Arm2: ({arm2_pos['start']['x']:.3f}, {arm2_pos['start']['y']:.3f}, {arm2_pos['start']['z']:.3f})")
        print(f"     → ({arm2_pos['target']['x']:.3f}, {arm2_pos['target']['y']:.3f}, {arm2_pos['target']['z']:.3f})")
        
        # Create position config file
        position_file = self.create_position_input_file(trial_num, arm1_pos, arm2_pos)
        
        # Initialize trial result
        trial_result = {
            'trial_number': trial_num,
            'timestamp': datetime.now().isoformat(),
            'positions': {
                'arm1': arm1_pos,
                'arm2': arm2_pos
            },
            'steps': {},
            'status': 'running'
        }
        
        # Step 1: IK Solver
        success, output, data = self.run_pipeline_step(
            'ik_solver',
            ['ros2', 'run', 'dual_arm_sync', 'dual_arm_ik_solver'],
            timeout=60
        )
        
        trial_result['steps']['ik_solver'] = {
            'success': success,
            'has_output_file': os.path.exists('ik_solutions.json')
        }
        
        if not success or not os.path.exists('ik_solutions.json'):
            trial_result['status'] = 'failed_at_ik'
            return self._finalize_trial(trial_num, trial_result, trial_start_time)
        
        # Step 2: Trajectory Generation
        success, output, data = self.run_pipeline_step(
            'trajectory_generation',
            ['ros2', 'run', 'dual_arm_sync', 'trajectory_generation'],
            timeout=60
        )
        
        trial_result['steps']['trajectory_generation'] = {
            'success': success,
            'has_output_file': os.path.exists('trajectories.json')
        }
        
        if not success or not os.path.exists('trajectories.json'):
            trial_result['status'] = 'failed_at_trajectory'
            return self._finalize_trial(trial_num, trial_result, trial_start_time)
        
        # Step 3: Collision Checker
        success, output, data = self.run_pipeline_step(
            'collision_checker',
            ['ros2', 'run', 'dual_arm_sync', 'collision_checker'],
            timeout=60
        )
        
        has_collision = data.get('has_collision', False)
        
        trial_result['steps']['collision_checker'] = {
            'success': success,
            'has_collision': has_collision,
            'collision_count': data.get('collision_count', 0)
        }
        
        if not success:
            trial_result['status'] = 'failed_at_collision_check'
            return self._finalize_trial(trial_num, trial_result, trial_start_time)
        
        # Update statistics
        if has_collision:
            self.results['statistics']['collision_at_checker'] += 1
        else:
            self.results['statistics']['collision_free_at_checker'] += 1
        
        # Step 4: Kuramoto Synchronization
        success, output, data = self.run_pipeline_step(
            'kuramoto_synchronization',
            ['ros2', 'run', 'dual_arm_sync', 'kuramoto_synchronization'],
            timeout=180
        )
        
        collision_free = data.get('collision_free', False)
        
        trial_result['steps']['kuramoto_synchronization'] = {
            'success': success,
            'collision_free_path_found': collision_free,
            'iterations': data.get('iterations', 0),
            'has_output_file': os.path.exists('synchronized_trajectories.json')
        }
        
        if not success:
            trial_result['status'] = 'failed_at_kuramoto'
            return self._finalize_trial(trial_num, trial_result, trial_start_time)
        
        # Update resolution statistics
        if has_collision and collision_free:
            self.results['statistics']['resolved_by_kuramoto'] += 1
            trial_result['status'] = 'collision_resolved'
        elif has_collision and not collision_free:
            self.results['statistics']['failed_after_kuramoto'] += 1
            trial_result['status'] = 'collision_unresolved'
        elif not has_collision:
            trial_result['status'] = 'collision_free'
        
        # Mark as successful execution if collision-free path exists
        if collision_free:
            self.results['statistics']['successful_executions'] += 1
        
        return self._finalize_trial(trial_num, trial_result, trial_start_time)
    
    def _finalize_trial(self, trial_num: int, trial_result: Dict, 
                       start_time: float) -> Dict:
        """Finalize trial, save results, backup files"""
        trial_result['duration'] = time.time() - start_time
        
        # Save trial result
        trial_file = self.output_dir / 'trials' / f"trial_{trial_num:03d}.json"
        with open(trial_file, 'w') as f:
            json.dump(trial_result, f, indent=2)
        
        # Backup trajectory files
        trial_traj_dir = self.output_dir / 'trajectories' / f"trial_{trial_num:03d}"
        trial_traj_dir.mkdir(exist_ok=True)
        
        for file in ['ik_solutions.json', 'trajectories.json', 'synchronized_trajectories.json']:
            if os.path.exists(file):
                shutil.copy(file, trial_traj_dir / file)
                os.remove(file)  # Clean up for next trial
        
        # Update statistics
        self.results['statistics']['completed_trials'] += 1
        
        print(f"\n✓ Trial {trial_num} completed: {trial_result['status']}")
        print(f"  Duration: {trial_result['duration']:.1f}s")
        
        return trial_result
    
    def print_progress(self):
        """Print current progress"""
        stats = self.results['statistics']
        
        print("\n" + "="*80)
        print("PROGRESS SUMMARY")
        print("="*80)
        print(f"Completed: {stats['completed_trials']}/{self.num_trials}")
        print(f"\nCollision Detection:")
        print(f"  ├─ Initial collisions: {stats['collision_at_checker']}")
        print(f"  └─ Collision-free: {stats['collision_free_at_checker']}")
        print(f"\nKuramoto Resolution:")
        print(f"  ├─ Resolved: {stats['resolved_by_kuramoto']}")
        print(f"  └─ Unresolved: {stats['failed_after_kuramoto']}")
        print(f"\nSuccess Rate: {stats['successful_executions']}/{stats['completed_trials']}", end='')
        
        if stats['completed_trials'] > 0:
            success_rate = stats['successful_executions'] / stats['completed_trials'] * 100
            print(f" ({success_rate:.1f}%)")
        else:
            print()
        
        print("="*80 + "\n")
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        stats = self.results['statistics']
        
        print("\n" + "="*80)
        print("FINAL REPORT - DUAL-ARM COLLISION AVOIDANCE TESTING")
        print("="*80)
        
        self.results['metadata']['end_time'] = datetime.now().isoformat()
        
        print(f"\nTest Period: {self.results['metadata']['start_time']}")
        print(f"         to: {self.results['metadata']['end_time']}")
        print(f"\nTotal Trials: {stats['completed_trials']}/{self.num_trials}")
        
        print(f"\n{'─'*80}")
        print("COLLISION ANALYSIS")
        print(f"{'─'*80}")
        print(f"\n1. Initial Collision Detection (collision_checker):")
        print(f"   Collisions detected: {stats['collision_at_checker']}")
        print(f"   Collision-free: {stats['collision_free_at_checker']}")
        
        if stats['completed_trials'] > 0:
            initial_collision_rate = stats['collision_at_checker'] / stats['completed_trials'] * 100
            print(f"   Collision rate: {initial_collision_rate:.1f}%")
        
        print(f"\n2. Kuramoto Synchronization Results:")
        print(f"   Collisions resolved: {stats['resolved_by_kuramoto']}")
        print(f"   Could not resolve: {stats['failed_after_kuramoto']}")
        
        if stats['collision_at_checker'] > 0:
            resolution_rate = stats['resolved_by_kuramoto'] / stats['collision_at_checker'] * 100
            print(f"   Resolution rate: {resolution_rate:.1f}%")
        
        print(f"\n3. Overall Success:")
        print(f"   Successful executions: {stats['successful_executions']}")
        print(f"   Failed trials: {stats['failed_trials']}")
        
        if stats['completed_trials'] > 0:
            overall_success = stats['successful_executions'] / stats['completed_trials'] * 100
            print(f"   Overall success rate: {overall_success:.1f}%")
        
        print(f"\n{'='*80}")
        
        # Save comprehensive report
        report_file = self.output_dir / "final_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_file}")
        
        # Generate summary files
        self._generate_summary_csv()
        self._generate_markdown_report()
    
    def _generate_summary_csv(self):
        """Generate CSV summary"""
        import csv
        
        csv_file = self.output_dir / "summary.csv"
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            writer.writerow([
                'Trial', 'Status', 'Duration_s',
                'IK_Success', 'Traj_Success', 'Collision_Detected',
                'Kuramoto_Success', 'Collision_Free_Path',
                'Arm1_Start_X', 'Arm1_Start_Y', 'Arm1_Start_Z',
                'Arm1_End_X', 'Arm1_End_Y', 'Arm1_End_Z',
                'Arm2_Start_X', 'Arm2_Start_Y', 'Arm2_Start_Z',
                'Arm2_End_X', 'Arm2_End_Y', 'Arm2_End_Z'
            ])
            
            for trial in self.results['trials']:
                a1 = trial['positions']['arm1']
                a2 = trial['positions']['arm2']
                steps = trial['steps']
                
                writer.writerow([
                    trial['trial_number'],
                    trial['status'],
                    f"{trial['duration']:.1f}",
                    steps.get('ik_solver', {}).get('success', False),
                    steps.get('trajectory_generation', {}).get('success', False),
                    steps.get('collision_checker', {}).get('has_collision', False),
                    steps.get('kuramoto_synchronization', {}).get('success', False),
                    steps.get('kuramoto_synchronization', {}).get('collision_free_path_found', False),
                    f"{a1['start']['x']:.3f}", f"{a1['start']['y']:.3f}", f"{a1['start']['z']:.3f}",
                    f"{a1['target']['x']:.3f}", f"{a1['target']['y']:.3f}", f"{a1['target']['z']:.3f}",
                    f"{a2['start']['x']:.3f}", f"{a2['start']['y']:.3f}", f"{a2['start']['z']:.3f}",
                    f"{a2['target']['x']:.3f}", f"{a2['target']['y']:.3f}", f"{a2['target']['z']:.3f}"
                ])
        
        print(f"CSV summary saved to: {csv_file}")
    
    def _generate_markdown_report(self):
        """Generate markdown report"""
        md_file = self.output_dir / "REPORT.md"
        stats = self.results['statistics']
        
        with open(md_file, 'w') as f:
            f.write("# Dual-Arm Collision Avoidance Test Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Total Trials:** {stats['completed_trials']}/{self.num_trials}\n")
            f.write(f"- **Successful Executions:** {stats['successful_executions']}\n")
            
            if stats['completed_trials'] > 0:
                success_rate = stats['successful_executions'] / stats['completed_trials'] * 100
                f.write(f"- **Success Rate:** {success_rate:.1f}%\n")
            
            f.write("\n## Collision Analysis\n\n")
            f.write("### Initial Detection (collision_checker)\n\n")
            f.write(f"- Collisions detected: **{stats['collision_at_checker']}**\n")
            f.write(f"- Collision-free: **{stats['collision_free_at_checker']}**\n")
            
            if stats['completed_trials'] > 0:
                collision_rate = stats['collision_at_checker'] / stats['completed_trials'] * 100
                f.write(f"- Collision rate: **{collision_rate:.1f}%**\n")
            
            f.write("\n### Kuramoto Resolution\n\n")
            f.write(f"- **Resolved:** {stats['resolved_by_kuramoto']}\n")
            f.write(f"- **Unresolved:** {stats['failed_after_kuramoto']}\n")
            
            if stats['collision_at_checker'] > 0:
                resolution_rate = stats['resolved_by_kuramoto'] / stats['collision_at_checker'] * 100
                f.write(f"- **Resolution Rate:** {resolution_rate:.1f}%\n")
            
            f.write("\n## Key Findings\n\n")
            
            if stats['collision_at_checker'] > 0:
                f.write(f"1. Out of {stats['completed_trials']} trials, ")
                f.write(f"{stats['collision_at_checker']} had collisions detected by the initial checker.\n\n")
                
                f.write(f"2. The Kuramoto synchronization successfully resolved ")
                f.write(f"{stats['resolved_by_kuramoto']} of these collisions.\n\n")
                
                f.write(f"3. {stats['failed_after_kuramoto']} collisions could not be resolved, ")
                f.write("indicating scenarios requiring further investigation.\n\n")
            
            f.write("\n## Files Generated\n\n")
            f.write("- `final_report.json` - Detailed results\n")
            f.write("- `summary.csv` - Trial-by-trial summary\n")
            f.write("- `trials/` - Individual trial results\n")
            f.write("- `trajectories/` - Generated trajectory files\n")
            f.write("- `logs/` - Execution logs\n")
        
        print(f"Markdown report saved to: {md_file}")
    
    def run_all_trials(self, strategy: str = 'random'):
        """Run all trials"""
        start_time = time.time()
        
        try:
            for trial_num in range(1, self.num_trials + 1):
                trial_result = self.run_single_trial(trial_num, strategy)
                self.results['trials'].append(trial_result)
                
                # Progress update every 10 trials
                if trial_num % 10 == 0 or trial_num == self.num_trials:
                    self.print_progress()
                
                # Small delay
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            print("\n\n⚠ Testing interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            elapsed = time.time() - start_time
            print(f"\nTotal time: {elapsed/60:.1f} minutes")
            self.generate_final_report()


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced dual-arm collision avoidance testing',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of trials (default: 100)')
    parser.add_argument('--output-dir', type=str, default='test_results',
                       help='Output directory (default: test_results)')
    parser.add_argument('--workspace-config', type=str, default=None,
                       help='Workspace configuration file (YAML)')
    parser.add_argument('--strategy', type=str, default='random',
                       choices=['random', 'systematic', 'challenging'],
                       help='Position generation strategy')
    
    args = parser.parse_args()
    
    orchestrator = EnhancedTestOrchestrator(
        num_trials=args.trials,
        output_dir=args.output_dir,
        workspace_config=args.workspace_config
    )
    
    orchestrator.run_all_trials(strategy=args.strategy)


if __name__ == '__main__':
    main()