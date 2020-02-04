import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default file names

# Locations used for data
# DATA_FOLDER = '/data2/data/d_hfe/'
# PROCESSING_FOLDER = '/data2/data/d_hfe/processing/'

DATA_FOLDER = '/dcache/alice/hezanoli/data/'
PROCESSING_FOLDER = '/dcache/alice/hezanoli/data/processing/'
TEMP = '/project/alice/users/hezanoli/temp/'  # used to save file before moving to dcache

# Cluster definitions
JOB_QUEUE = 'short7'
JOB_QUEUE_GPU = 'gpu7'
CLUSTER_MEMORY = 8

# treelite
TOOLCHAIN = 'gcc'
