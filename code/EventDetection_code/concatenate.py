import os

dataDir = '/home/beams0/WZHENG/032625_test_patches/'

listFiles = os.listdir(dataDir)

from src.datasets.ge2patch import *

newName = '../sam9_all_init.edf.h5'

test = concatenate_patches(dataDir, listFiles, newName)




