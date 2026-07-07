#!/usr/bin/env python3
"""
ik_interactive.py
Interactive IK Controller for Doosan M1013

Features:
- Works with both Gazebo and RViz
- Reads actual joint states (handles Gazebo joint ordering)
- Moves robot to target positions
- Publishes to correct topics for both simulators
- World/Local frame support

Usage:
    ros2 run dual_arm_sync ik_interactive
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker
import numpy as np

try:
    from dual_arm_sync.constants import DHParameters, JointLimits, LinkLengths
    from dual_arm_sync.ik_solver import forward_kinematics, solve_ik_numerical, compute_cost, select_optimal_solution, RobotBases
except ImportError:
    print("ERROR: Cannot import required modules")
    import sys
    sys.exit(1)


class IKInteractiveController(Node):
    """
    Interactive IK controller with visualization
    Reads joint states, solves IK, moves robot
    """
    
    def __init__(self, robot_name: str = 'dsr01'):
        super().__init__(f'{robot_name}_ik_interactive')
        
        self.robot_name = robot_name
        self.current_joints = np.zeros(6)
        self.joints_received = False
        
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        
        # Subscribe to joint states (both Gazebo and RViz)
        self.joint_sub_gz = self.create_subscription(
            JointState,
            f'/{robot_name}/gz/joint_states',
            self.joint_callback,
            10
        )
        
        self.joint_sub_rviz = self.create_subscription(
            JointState,
            f'/{robot_name}/joint_states',
            self.joint_callback,
            10
        )
        
        # Publish joint states (for RViz)
        self.joint_pub = self.create_publisher(
            JointState,
            f'/{robot_name}/joint_states',
            10
        )
        
        # Publish joint commands (for Gazebo)
        self.gazebo_cmd_pub = self.create_publisher(
            Float64MultiArray,
            f'/{robot_name}/gz/dsr_position_controller/commands',
            10
        )
        
        # Publish target marker
        self.marker_pub = self.create_publisher(
            Marker,
            '/ik_target_marker',
            10
        )
        
        # Timer for publishing
        self.timer = self.create_timer(0.05, self.publish_joint_states)
        
        self.get_logger().info('='*70)
        self.get_logger().info(f'IK Interactive Controller Started: {robot_name}')
        self.get_logger().info(f'Robot base: {RobotBases.get_base_position(robot_name)}')
        self.get_logger().info('='*70)
    
    def joint_callback(self, msg: JointState):
        """
        Update current joint configuration
        Handles Gazebo's incorrect joint ordering
        """
        if len(msg.position) < 6:
            return
        
        # Create name to index mapping
        joint_map = {}
        for i, name in enumerate(msg.name):
            joint_map[name] = i
        
        # Reorder to [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]
        if all(f'joint_{i+1}' in joint_map for i in range(6)):
            self.current_joints = np.array([
                msg.position[joint_map['joint_1']],
                msg.position[joint_map['joint_2']],
                msg.position[joint_map['joint_3']],
                msg.position[joint_map['joint_4']],
                msg.position[joint_map['joint_5']],
                msg.position[joint_map['joint_6']],
            ])
        else:
            self.current_joints = np.array(msg.position[:6])
        
        if not self.joints_received:
            self.get_logger().info('✓ Joint states received')
            self.get_logger().info(f'  Current (deg): {np.degrees(self.current_joints)}')
            
            # Show current end-effector position
            fk_pos, _ = forward_kinematics(self.current_joints)
            world_pos = RobotBases.local_to_world(fk_pos, self.robot_name)
            self.get_logger().info(f'  End-effector local:  {fk_pos}')
            self.get_logger().info(f'  End-effector world:  {world_pos}')
            
            self.joints_received = True
    
    def move_to_position(self, target_pos: np.ndarray, use_world_frame: bool = True):
        """
        Solve IK and move robot to target position
        
        Args:
            target_pos: [x, y, z] target position
            use_world_frame: If True, target is in world frame
        """
        if not self.joints_received:
            self.get_logger().warn('⚠️  Waiting for joint states...')
            return False
        
        # Convert to local frame
        if use_world_frame:
            local_target = RobotBases.world_to_local(target_pos, self.robot_name)
            self.get_logger().info(f'\nTarget (world): {target_pos}')
            self.get_logger().info(f'Target (local): {local_target}')
        else:
            local_target = target_pos
            self.get_logger().info(f'\nTarget (local): {target_pos}')
        
        self.get_logger().info(f'Current joints (deg): {np.degrees(self.current_joints)}')
        self.get_logger().info('Solving IK...')
        
        # Solve IK
        solutions = solve_ik_numerical(local_target, self.current_joints)
        
        if len(solutions) == 0:
            self.get_logger().error('❌ No IK solutions found!')
            return False
        
        self.get_logger().info(f'✓ Found {len(solutions)} solutions')
        
        # Select optimal
        result = select_optimal_solution(solutions, self.current_joints, verbose=False)
        
        if result is None:
            self.get_logger().error('❌ Failed to select solution')
            return False
        
        optimal_joints, info = result
        
        self.get_logger().info(f'✓ Optimal solution: cost={info["cost"]:.4f}')
        self.get_logger().info(f'  Target joints (deg): {info["optimal_joints_deg"]}')
        self.get_logger().info(f'  Displacement (deg): {info["displacement_deg"]}')
        
        # Verify
        fk_pos, _ = forward_kinematics(optimal_joints)
        error = np.linalg.norm(fk_pos - local_target)
        self.get_logger().info(f'  Position error: {error*1000:.2f} mm')
        
        # IMPORTANT: Update target joints (not current - let joint_callback update current)
        # This will be published by the timer
        self.current_joints = optimal_joints
        
        # Publish target marker
        self.publish_target_marker(target_pos if use_world_frame else 
                                   RobotBases.local_to_world(target_pos, self.robot_name))
        
        # Give time for robot to start moving
        self.get_logger().info('Moving robot...')
        import time
        time.sleep(0.5)  # Allow some publishing cycles
        
        return True
    
    def publish_joint_states(self):
        """Publish joint states to both RViz and Gazebo"""
        # For RViz
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.current_joints.tolist()
        self.joint_pub.publish(msg)
        
        # For Gazebo
        gazebo_msg = Float64MultiArray()
        gazebo_msg.data = self.current_joints.tolist()
        self.gazebo_cmd_pub.publish(gazebo_msg)
    
    def publish_target_marker(self, world_position: np.ndarray):
        """Visualize target in RViz"""
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'ik_targets'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(world_position[0])
        marker.pose.position.y = float(world_position[1])
        marker.pose.position.z = float(world_position[2])
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        self.marker_pub.publish(marker)
    
    def interactive_loop(self):
        """Interactive command loop"""
        print("\n" + "="*70)
        print("INTERACTIVE IK CONTROLLER")
        print("="*70)
        print("\nCommands:")
        print("  1-9  : Preset positions")
        print("  c    : Custom position")
        print("  h    : Home position")
        print("  s    : Show current state")
        print("  t    : Test movement (move joint 3 to -45°)")
        print("  q    : Quit")
        print("="*70 + "\n")
        
        # Wait for joint states
        print("Waiting for joint states...")
        while not self.joints_received and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if not self.joints_received:
            print("❌ Failed to receive joint states!")
            return
        
        print("✓ Ready!\n")
        
        # Preset positions (LOCAL frame)
        presets = {
            '1': (0.5, 0.0, 0.8, "Front center"),
            '2': (0.4, 0.2, 0.6, "Right side"),
            '3': (0.4, -0.2, 0.6, "Left side"),
            '4': (0.6, 0.0, 0.5, "Extended front"),
            '5': (0.3, 0.0, 1.0, "High front"),
            '6': (0.5, 0.3, 0.7, "Extended right"),
            '7': (0.5, -0.3, 0.7, "Extended left"),
            '8': (0.4, 0.0, 0.4, "Low front"),
            '9': (0.3, 0.0, 0.6, "Medium front"),
        }
        
        while rclpy.ok():
            try:
                print("\nCommand: ", end='', flush=True)
                command = input().strip().lower()
                
                if command == 'q':
                    print("Exiting...")
                    break
                
                elif command == 'h':
                    print("Moving to home...")
                    self.current_joints = np.zeros(6)
                    print("✓ At home")
                
                elif command == 't':
                    print("Testing movement: Moving joint 3 to -45°...")
                    self.current_joints = np.array([0, 0, np.radians(-45), 0, 0, 0])
                    print("✓ Test command sent")
                    print("  Watch Gazebo for movement!")
                
                elif command == 's':
                    fk_pos, _ = forward_kinematics(self.current_joints)
                    world_pos = RobotBases.local_to_world(fk_pos, self.robot_name)
                    print(f"\nCurrent state:")
                    print(f"  Joints (deg): {np.degrees(self.current_joints)}")
                    print(f"  End-effector local:  {fk_pos}")
                    print(f"  End-effector world:  {world_pos}")
                
                elif command in presets:
                    x, y, z, name = presets[command]
                    print(f"Moving to: {name}")
                    # Use local frame for presets
                    self.move_to_position(np.array([x, y, z]), use_world_frame=False)
                
                elif command == 'c':
                    try:
                        print("\nCoordinate frame:")
                        print("  1. World frame")
                        print("  2. Local frame")
                        frame = input("Frame (1/2): ").strip()
                        use_world = frame != '2'
                        
                        x = float(input("X (m): "))
                        y = float(input("Y (m): "))
                        z = float(input("Z (m): "))
                        
                        self.move_to_position(np.array([x, y, z]), use_world_frame=use_world)
                    except ValueError:
                        print("Invalid input!")
                
                else:
                    print("Unknown command")
                
                rclpy.spin_once(self, timeout_sec=0.01)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                self.get_logger().error(f"Error: {e}")


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    print("\n" + "="*80)
    print("IK INTERACTIVE CONTROLLER")
    print("="*80)
    print("\nSelect robot:")
    print("  1. dsr01 (base at y=0.5)")
    print("  2. dsr02 (base at y=-0.5)")
    
    try:
        choice = input("\nChoice (1/2): ").strip()
        robot_name = 'dsr02' if choice == '2' else 'dsr01'
    except:
        robot_name = 'dsr01'
    
    try:
        controller = IKInteractiveController(robot_name)
        controller.interactive_loop()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        print("\nShutdown requested")
    finally:
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()