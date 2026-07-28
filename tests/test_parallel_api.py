
import sys

sys.path.insert(0,'src')

from environment import Environment
from pettingzoo.test import parallel_api_test


parallel_api_test(Environment(15, 15, robot_ids=['r0','r1','r2'], obstacle_density=0.2), num_cycles=2000)
