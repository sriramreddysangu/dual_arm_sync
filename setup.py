from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'dual_arm_sync'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
        ('share/' + package_name + '/worlds', glob('worlds/*.world')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dual Arm Team',
    maintainer_email='me24s501@iittp.ac.in',
    description='Dual-arm trajectory generation with collision detection',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Configuration
            'test_constants = dual_arm_sync.constants:print_config_summary',
            
            # IK Solvers
            'ik_solver = dual_arm_sync.ik_solver:main',
            'test_ik = dual_arm_sync.ik_solver:test_ik_solver_standalone',
            'dual_arm_ik_solver = dual_arm_sync.dual_arm_ik_solver:main',
            'ik_interactive = dual_arm_sync.ik_interactive:main',
            
            # Trajectory Generation (reads ik_solutions.json)
            'trajectory_generation = dual_arm_sync.trajectory_generation:main',
            'test_trajectory = dual_arm_sync.trajectory_generation:test_trajectory_generation',
            
            # Collision Detection (reads trajectories.json)
            'collision_checker = dual_arm_sync.collision_checker:main',
            
            # Kuramoto Synchronization (reads collision_report.json)
            'kuramoto_synchronization = dual_arm_sync.kuramoto_synchronization:main',
            'kuramoto_sync = dual_arm_sync.kuramoto_sync:main',
            
            # RRT-Connect Planner (NEW! - reads ik_solutions.json and synchronized_trajectories.json)
            'rrt_connect_planner = dual_arm_sync.rrt_connect_planner:main',
            
            # Execution
            'gazebo_executor = dual_arm_sync.gazebo_executor:main',
            
            # Testing
            'results_step1 = dual_arm_sync.results_step1:main',
            
            'quad_arm_config           = dual_arm_sync.quad_arm_config:print_config',

            # Step 1 — IK
            'quad_arm_ik_solver        = dual_arm_sync.quad_arm_ik_solver:main',

            # Step 2 — Trajectory generation
            'quad_trajectory_generation = dual_arm_sync.quad_trajectory_generation:main',

            # Step 3 — Collision checking
            'quad_collision_checker    = dual_arm_sync.quad_collision_checker:main',

            # Step 4a — Kuramoto synchronization (collision resolver)
            'quad_kuramoto_synchronization = dual_arm_sync.quad_kuramoto_synchronization:main',

            # Step 4b — RRT planner (fallback when Kuramoto fails)
            'quad_rrt_planner          = dual_arm_sync.quad_rrt_planner:main',

            # Step 5 — Gazebo executor
            
            'quad_gazebo_executor      = dual_arm_sync.quad_gazebo_executor:main',

            # Utilities
            'quad_arm_test_pub         = dual_arm_sync.quad_arm_test_pub:main',
            'quad_arm_results          = dual_arm_sync.quad_arm_results:main',

            'local_deformation = dual_arm_sync.local_deformation:main', 
            'multi_arm_pipeline = dual_arm_sync.multi_arm_trajectory_planner:run_demo',


            'step_1 = dual_arm_sync.step_1:main',
            'step_2 = dual_arm_sync.step_2:main',
            'step_3 = dual_arm_sync.step_3:main',
            'step_4 = dual_arm_sync.step_4:main',
            'step_5 = dual_arm_sync.step_5:main',
            'step_6 = dual_arm_sync.step_6:main',

            'step_7 = dual_arm_sync.step_7:main',
            'step_8 = dual_arm_sync.step_8:main',
            'step_9 = dual_arm_sync.step_9:main',
            'step_10 = dual_arm_sync.step_10:main',


            'mesh_collision = dual_arm_sync.mesh_collision:main',
            'step_11 = dual_arm_sync.step_11:main',
            'step_12 = dual_arm_sync.step_12:main',
            'step_13 = dual_arm_sync.step_13:main',
            'step_14 = dual_arm_sync.step_14:main',
            'step_15 = dual_arm_sync.step_15:main',
            'step_16 = dual_arm_sync.step_16:main',
            'step_21 = dual_arm_sync.step_21:main',
            'step_22 = dual_arm_sync.step_22:main',
            'step_23 = dual_arm_sync.step_23:main',
            'step_24 = dual_arm_sync.step_24:main',
            'step_25 = dual_arm_sync.step_25:main',
            'step_26 = dual_arm_sync.step_26:main',
            'step_27 = dual_arm_sync.step_27:main',
            '_robot4x = dual_arm_sync._robot4x:main',
            'step_31 = dual_arm_sync.step_31:main',
            'step_32 = dual_arm_sync.step_32:main',
            'step_33 = dual_arm_sync.step_33:main',
            'step_34 = dual_arm_sync.step_34:main',
            'step_34_diag = dual_arm_sync.step_34_diag:main',
            'step_35 = dual_arm_sync.step_35:main',
            'step_40 = dual_arm_sync.step_40:main',
            'step_41 = dual_arm_sync.step_41:main',
            'step_42 = dual_arm_sync.step_42:main',
            'step_43 = dual_arm_sync.step_43:main',
            'step_44 = dual_arm_sync.step_44:main',
            'step_45 = dual_arm_sync.step_45:main',
            'step_46 = dual_arm_sync.step_46:main',
            '_robot = dual_arm_sync._robot:main',
            'step_51 = dual_arm_sync.step_51:main',
            'step_52 = dual_arm_sync.step_52:main',
            'step_53 = dual_arm_sync.step_53:main',
            'step_54 = dual_arm_sync.step_54:main',

            'step_55 = dual_arm_sync.step_55:main',
            'step_56 = dual_arm_sync.step_56:main',
            'step_57 = dual_arm_sync.step_57:main',
            'step_61 = dual_arm_sync.step_61:main',
            'step_62 = dual_arm_sync.step_62:main',
            'step_63 = dual_arm_sync.step_63:main',
            'step_64 = dual_arm_sync.step_64:main',
            'step_64_viz = dual_arm_sync.step_64_viz:main',
            'step_65 = dual_arm_sync.step_65:main',
            'step_66 = dual_arm_sync.step_66:main',
            'step_67 = dual_arm_sync.step_67:main',
            'step_68 = dual_arm_sync.step_68:main',
            'step_69 = dual_arm_sync.step_69:main',
            'step_70 = dual_arm_sync.step_70:main',
            'step_71 = dual_arm_sync.step_71:main',
            'step_72 = dual_arm_sync.step_72:main',
            'step_73 = dual_arm_sync.step_73:main',
            'step_74 = dual_arm_sync.step_74:main',
            'step_75 = dual_arm_sync.step_75:main',
            'step_76 = dual_arm_sync.step_76:main',
            'step_77 = dual_arm_sync.step_77:main',
            'step_78 = dual_arm_sync.step_78:main',
            'step_79 = dual_arm_sync.step_79:main',
            'step_80 = dual_arm_sync.step_80:main',
            'run_pipeline = dual_arm_sync.run_pipeline:main',
            'step_100 = dual_arm_sync.step_100:main',
            'step_81 = dual_arm_sync.step_81:main',
            'step_82 = dual_arm_sync.step_82:main',
            'step_83 = dual_arm_sync.step_83:main',







            'bench_core = dual_arm_sync.bench_core:main',
            'raw_dual_move = dual_arm_sync.raw_dual_move:main',
            'step_plan = dual_arm_sync.step_plan:main',
            'multi_arm_core = dual_arm_sync.multi_arm_core:main',

        ],
    },
)