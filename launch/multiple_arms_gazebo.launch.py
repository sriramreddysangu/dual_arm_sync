#!/usr/bin/env python3
"""
multi_arm_gazebo.launch.py - Multi Arm Gazebo Launch (FIXED - No duplicate spawning)
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

        # Robot3 args
        DeclareLaunchArgument('name3', default_value='dsr03', description='Robot3 namespace'),
        DeclareLaunchArgument('color3', default_value='blue', description='Robot3 color'),
        DeclareLaunchArgument('model3', default_value='m1013', description='Robot3 model'),
        DeclareLaunchArgument('x3', default_value='1.0', description='Robot3 x pose'),
        DeclareLaunchArgument('y3', default_value='0.5', description='Robot3 y pose'),
        DeclareLaunchArgument('z3', default_value='0.0', description='Robot3 z pose'),
        DeclareLaunchArgument('R3', default_value='0.0', description='Robot3 roll'),
        DeclareLaunchArgument('P3', default_value='0.0', description='Robot3 pitch'),
        DeclareLaunchArgument('Y3', default_value='0.0', description='Robot3 yaw'),

        # Robot4 args
        DeclareLaunchArgument('name4', default_value='dsr04', description='Robot4 namespace'),
        DeclareLaunchArgument('color4', default_value='white', description='Robot4 color'),
        DeclareLaunchArgument('model4', default_value='m1013', description='Robot4 model'),
        DeclareLaunchArgument('x4', default_value='1.0', description='Robot4 x pose'),
        DeclareLaunchArgument('y4', default_value='-0.5', description='Robot4 y pose'),
        DeclareLaunchArgument('z4', default_value='0.0', description='Robot4 z pose'),
        DeclareLaunchArgument('R4', default_value='0.0', description='Robot4 roll'),
        DeclareLaunchArgument('P4', default_value='0.0', description='Robot4 pitch'),
        DeclareLaunchArgument('Y4', default_value='0.0', description='Robot4 yaw'),


        # Robot5 args
        DeclareLaunchArgument('name5', default_value='dsr05', description='Robot5 namespace'),
        DeclareLaunchArgument('color5', default_value='blue', description='Robot5 color'),
        DeclareLaunchArgument('model5', default_value='m1013', description='Robot5 model'),
        DeclareLaunchArgument('x5', default_value='-1.0', description='Robot5 x pose'),
        DeclareLaunchArgument('y5', default_value='0.5', description='Robot5 y pose'),
        DeclareLaunchArgument('z5', default_value='0.0', description='Robot5 z pose'),
        DeclareLaunchArgument('R5', default_value='0.0', description='Robot5 roll'),
        DeclareLaunchArgument('P5', default_value='0.0', description='Robot5 pitch'),
        DeclareLaunchArgument('Y5', default_value='0.0', description='Robot5 yaw'),

        # Robot6 args
        DeclareLaunchArgument('name6', default_value='dsr06', description='Robot6 namespace'),
        DeclareLaunchArgument('color6', default_value='white', description='Robot6 color'),
        DeclareLaunchArgument('model6', default_value='m1013', description='Robot6 model'),
        DeclareLaunchArgument('x6', default_value='-1.0', description='Robot6 x pose'),
        DeclareLaunchArgument('y6', default_value='-0.5', description='Robot6 y pose'),
        DeclareLaunchArgument('z6', default_value='0.0', description='Robot6 z pose'),
        DeclareLaunchArgument('R6', default_value='0.0', description='Robot6 roll'),
        DeclareLaunchArgument('P6', default_value='0.0', description='Robot6 pitch'),
        DeclareLaunchArgument('Y6', default_value='0.0', description='Robot6 yaw'),

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


    # Spawn Robot 3 using dsr_spawn_on_gazebo.launch.py
    robot3_launch = TimerAction(
        period=13.0,  # Wait for Gazebo to start
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('dsr_gazebo2'), 
                            'launch', 'dsr_spawn_on_gazebo.launch.py')
            ),
            launch_arguments={
                'use_gazebo': 'true',
                'name': LaunchConfiguration('name3'),
                'model': LaunchConfiguration('model3'),
                'color': LaunchConfiguration('color3'),
                'x': LaunchConfiguration('x3'),
                'y': LaunchConfiguration('y3'),
                'z': LaunchConfiguration('z3'),
                'R': LaunchConfiguration('R3'),
                'P': LaunchConfiguration('P3'),
                'Y': LaunchConfiguration('Y3'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'remap_tf': 'false',
            }.items(),
        )]
    )

    # Spawn Robot 4 using dsr_spawn_on_gazebo.launch.py (delayed)
    robot4_launch = TimerAction(
        period=18.0,  # Wait for Robot 1 to fully spawn
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('dsr_gazebo2'), 
                            'launch', 'dsr_spawn_on_gazebo.launch.py')
            ),
            launch_arguments={
                'use_gazebo': 'true',
                'name': LaunchConfiguration('name4'),
                'model': LaunchConfiguration('model4'),
                'color': LaunchConfiguration('color4'),
                'x': LaunchConfiguration('x4'),
                'y': LaunchConfiguration('y4'),
                'z': LaunchConfiguration('z4'),
                'R': LaunchConfiguration('R4'),
                'P': LaunchConfiguration('P4'),
                'Y': LaunchConfiguration('Y4'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'remap_tf': 'false',
            }.items(),
        )]
    )


        # Spawn Robot 5 using dsr_spawn_on_gazebo.launch.py
    robot5_launch = TimerAction(
        period=23.0,  # Wait for Gazebo to start
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('dsr_gazebo2'), 
                            'launch', 'dsr_spawn_on_gazebo.launch.py')
            ),
            launch_arguments={
                'use_gazebo': 'true',
                'name': LaunchConfiguration('name5'),
                'model': LaunchConfiguration('model5'),
                'color': LaunchConfiguration('color5'),
                'x': LaunchConfiguration('x5'),
                'y': LaunchConfiguration('y5'),
                'z': LaunchConfiguration('z5'),
                'R': LaunchConfiguration('R5'),
                'P': LaunchConfiguration('P5'),
                'Y': LaunchConfiguration('Y5'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'remap_tf': 'false',
            }.items(),
        )]
    )

    # Spawn Robot 6 using dsr_spawn_on_gazebo.launch.py (delayed)
    robot6_launch = TimerAction(
        period=28.0,  # Wait for Robot 1 to fully spawn
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('dsr_gazebo2'), 
                            'launch', 'dsr_spawn_on_gazebo.launch.py')
            ),
            launch_arguments={
                'use_gazebo': 'true',
                'name': LaunchConfiguration('name6'),
                'model': LaunchConfiguration('model6'),
                'color': LaunchConfiguration('color6'),
                'x': LaunchConfiguration('x6'),
                'y': LaunchConfiguration('y6'),
                'z': LaunchConfiguration('z6'),
                'R': LaunchConfiguration('R6'),
                'P': LaunchConfiguration('P6'),
                'Y': LaunchConfiguration('Y6'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'remap_tf': 'false',
            }.items(),
        )]
    )



    return LaunchDescription(ARGUMENTS + [
        gazebo,
        robot1_launch,
        robot2_launch,
        robot3_launch,
        robot4_launch,
        robot5_launch,
        robot6_launch,
    ])