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
            







            'bench_core = dual_arm_sync.bench_core:main',
            'raw_dual_move = dual_arm_sync.raw_dual_move:main',
            'step_plan = dual_arm_sync.step_plan:main',
            'multi_arm_core = dual_arm_sync.multi_arm_core:main',

        ],
    },
)
