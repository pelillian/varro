"""
This module stores the static variables we'll need for the project
"""

import os 


ROOT_DIR = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-2]) # This is the Project Root
ABSOLUTE_ALGO_LOGS_PATH = os.path.join(ROOT_DIR, 'logs/varro/algo')