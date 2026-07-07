#!/usr/bin/env python3
"""
dual_gazebo.launch.py - Dual Arm Gazebo Launch (FIXED - No duplicate spawning)
Launches Gazebo once and spawns two robots using dsr_spawn_on_gazebo.launch.py
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    # Launch Arguments
    ARGUMENTS = [
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation clock'),
        
        # Robot1 args
        DeclareLaunchArgument('name1', default_value='dsr01', description='Robot1 namespace'),
        DeclareLaunchArgument('color1', default_value='white', description='Robot1 color'),
        DeclareLaunchArgument('model1', default_value='m1013', description='Robot1 model'),
        DeclareLaunchArgument('x1', default_value='0.0', description='Robot1 x pose'),
        DeclareLaunchArgument('y1', default_value='0.5', description='Robot1 y pose'),
        DeclareLaunchArgument('z1', default_value='0.0', description='Robot1 z pose'),
        DeclareLaunchArgument('R1', default_value='0.0', description='Robot1 roll'),
        DeclareLaunchArgument('P1', default_value='0.0', description='Robot1 pitch'),
        DeclareLaunchArgument('Y1', default_value='0.0', description='Robot1 yaw'),
        
        # Robot2 args
        DeclareLaunchArgument('name2', default_value='dsr02', description='Robot2 namespace'),
        DeclareLaunchArgument('color2', default_value='blue', description='Robot2 color'),
        DeclareLaunchArgument('model2', default_value='m1013', description='Robot2 model'),
        DeclareLaunchArgument('x2', default_value='0.0', description='Robot2 x pose'),
        DeclareLaunchArgument('y2', default_value='-0.5', description='Robot2 y pose'),
        DeclareLaunchArgument('z2', default_value='0.0', description='Robot2 z pose'),
        DeclareLaunchArgument('R2', default_value='0.0', description='Robot2 roll'),
        DeclareLaunchArgument('P2', default_value='0.0', description='Robot2 pitch'),
        DeclareLaunchArgument('Y2', default_value='0.0', description='Robot2 yaw'),

    ]

    # Launch Gazebo ONLY (no robot spawning from dsr_gazebo.launch.py)
    # Use the low-level gz_sim.launch.py directly
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"])
        ),
        launch_arguments={"gz_args": " -r -v 3 empty.sdf"}.items(),
    )

    # Spawn Robot 1 using dsr_spawn_on_gazebo.launch.py
    robot1_launch = TimerAction(
        period=3.0,  # Wait for Gazebo to start
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('dsr_gazebo2'), 
                            'launch', 'dsr_spawn_on_gazebo.launch.py')
            ),
            launch_arguments={
                'use_gazebo': 'true',
                'name': LaunchConfiguration('name1'),
                'model': LaunchConfiguration('model1'),
                'color': LaunchConfiguration('color1'),
                'x': LaunchConfiguration('x1'),
                'y': LaunchConfiguration('y1'),
                'z': LaunchConfiguration('z1'),
                'R': LaunchConfiguration('R1'),
                'P': LaunchConfiguration('P1'),
                'Y': LaunchConfiguration('Y1'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'remap_tf': 'false',
            }.items(),
        )]
    )

    # Spawn Robot 2 using dsr_spawn_on_gazebo.launch.py (delayed)
    robot2_launch = TimerAction(
        period=8.0,  # Wait for Robot 1 to fully spawn
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('dsr_gazebo2'), 
                            'launch', 'dsr_spawn_on_gazebo.launch.py')
            ),
            launch_arguments={
                'use_gazebo': 'true',
                'name': LaunchConfiguration('name2'),
                'model': LaunchConfiguration('model2'),
                'color': LaunchConfiguration('color2'),
                'x': LaunchConfiguration('x2'),
                'y': LaunchConfiguration('y2'),
                'z': LaunchConfiguration('z2'),
                'R': LaunchConfiguration('R2'),
                'P': LaunchConfiguration('P2'),
                'Y': LaunchConfiguration('Y2'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'remap_tf': 'false',
            }.items(),
        )]
    )



    return LaunchDescription(ARGUMENTS + [
        gazebo,
        robot1_launch,
        robot2_launch,
    ])