#!/usr/bin/env python3
"""
dual_rviz_fixed.launch.py - Fixed Dual Arm RViz Launch
Correctly displays two Doosan robots in RViz with proper positioning.
Key fix: Connect to each robot's internal 'world' frame, not base_link
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare all launch arguments
    ARGUMENTS = [
        DeclareLaunchArgument('use_sim_time', default_value='false', description='Use simulation clock'),
        
        # Robot 1
        DeclareLaunchArgument('name1', default_value='dsr01', description='Robot1 namespace'),
        DeclareLaunchArgument('color1', default_value='white', description='Robot1 color'),
        DeclareLaunchArgument('model1', default_value='m1013', description='Robot1 model'),
        DeclareLaunchArgument('x1', default_value='0.0', description='Robot1 x pose'),
        DeclareLaunchArgument('y1', default_value='0.5', description='Robot1 y pose'),
        DeclareLaunchArgument('z1', default_value='0.0', description='Robot1 z pose'),
        DeclareLaunchArgument('yaw1', default_value='0.0', description='Robot1 yaw'),
        
        # Robot 2
        DeclareLaunchArgument('name2', default_value='dsr02', description='Robot2 namespace'),
        DeclareLaunchArgument('color2', default_value='blue', description='Robot2 color'),
        DeclareLaunchArgument('model2', default_value='m1013', description='Robot2 model'),
        DeclareLaunchArgument('x2', default_value='0.0', description='Robot2 x pose'),
        DeclareLaunchArgument('y2', default_value='-0.5', description='Robot2 y pose'),
        DeclareLaunchArgument('z2', default_value='0.0', description='Robot2 z pose'),
        DeclareLaunchArgument('yaw2', default_value='0.0', description='Robot2 yaw (facing robot1)'),
    ]
    
    # Path to xacro files
    xacro_path = os.path.join(
        get_package_share_directory('dsr_description2'),
        'xacro'
    )
    
    # RViz configuration - try dual_arm.rviz, fallback to default
    rviz_config_file = os.path.join(
        get_package_share_directory('dsr_description2'),
        'rviz',
        'default.rviz'
    )
    
    # Try to find dual_arm specific config
    try:
        dual_arm_rviz = os.path.join(
            get_package_share_directory('dual_arm_sync'),
            'rviz',
            'dual_arm.rviz'
        )
        if os.path.exists(dual_arm_rviz):
            rviz_config_file = dual_arm_rviz
    except:
        pass
    
    # Robot 1 State Publisher
    robot1_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=LaunchConfiguration('name1'),
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'robot_description': Command([
                'xacro ', xacro_path, '/', LaunchConfiguration('model1'), '.urdf.xacro ',
                'color:=', LaunchConfiguration('color1'), ' ',
                'gripper:=none'
            ]),
            'frame_prefix': [LaunchConfiguration('name1'), '/'],
        }]
    )
    
    # Robot 2 State Publisher  
    robot2_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=LaunchConfiguration('name2'),
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'robot_description': Command([
                'xacro ', xacro_path, '/', LaunchConfiguration('model2'), '.urdf.xacro ',
                'color:=', LaunchConfiguration('color2'), ' ',
                'gripper:=none'
            ]),
            'frame_prefix': [LaunchConfiguration('name2'), '/'],
        }]
    )
    
    # CRITICAL FIX: Connect to each robot's internal 'world' frame
    # Static TF: world -> dsr01/world (not base_link!)
    static_tf_robot1 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_dsr01',
        arguments=[
            LaunchConfiguration('x1'),
            LaunchConfiguration('y1'),
            LaunchConfiguration('z1'),
            LaunchConfiguration('yaw1'),
            '0.0',  # pitch
            '0.0',  # roll
            'world',
            [LaunchConfiguration('name1'), '/world']  # Connect to dsr01/world
        ]
    )
    
    # Static TF: world -> dsr02/world (not base_link!)
    static_tf_robot2 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_dsr02',
        arguments=[
            LaunchConfiguration('x2'),
            LaunchConfiguration('y2'),
            LaunchConfiguration('z2'),
            LaunchConfiguration('yaw2'),
            '0.0',  # pitch
            '0.0',  # roll
            'world',
            [LaunchConfiguration('name2'), '/world']  # Connect to dsr02/world
        ]
    )
    
    # Joint State Publisher for Robot 1 (optional - for moving joints manually)
    joint_state_pub_gui_1 = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        namespace=LaunchConfiguration('name1'),
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )
    
    # Joint State Publisher for Robot 2 (optional)
    joint_state_pub_gui_2 = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        namespace=LaunchConfiguration('name2'),
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )
    
    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )
    
    return LaunchDescription(ARGUMENTS + [
        robot1_state_pub,
        robot2_state_pub,
        static_tf_robot1,
        static_tf_robot2,
        joint_state_pub_gui_1,
        joint_state_pub_gui_2,
        rviz_node
    ])
