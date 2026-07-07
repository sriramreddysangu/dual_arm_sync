"""
dual_arm_sync
Synchronized dual-arm trajectory generation for Doosan M1013 robots

Implements hierarchical collision detection with RRT-Connect warm-start
and B-spline trajectory optimization.

Based on: "Generation of Synchronized Configuration Space Trajectories 
of Multi-Robot Systems" by Ariyan Kabir
"""

__version__ = '1.0.0'
__author__ = 'sriram'
__email__ = 'me24s501@iittp.ac.in'

# Import main components

from . import constants

__all__ = [
    'constants',
]