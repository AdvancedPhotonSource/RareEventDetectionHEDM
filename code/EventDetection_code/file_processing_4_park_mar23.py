#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os

dataDir = '/net/wolf/data/tomo1/park_mar22_data/park_mar22/ge5/'
# dataDir = '/lambda_stor/homes/wzheng/raw_park22/'
# dataDir = '/lambda_stor/homes/wzheng/test_raw/'

listFiles = os.listdir(dataDir)


darkFilesList = []
darkFilesDict = {}

for fileString in listFiles:
    if fileString.startswith('dark_before'):
        # find the index of each dark before
        idx = int(fileString[12:18])
        #print(int(fileString[12:18]))
        darkFilesDict[idx] = fileString



darkFilesKeys = sorted(darkFilesDict)


parkFilesList = []
parkFilesDict4Idx = {}

# park_ss_ff

for fileString in listFiles:
    if fileString.startswith('park_ss_ff'):
        # find the index of each file
        x = fileString.split("_")
        idx = int(x[-1].split(".")[0])
        parkFilesDict4Idx[fileString] = idx
#         idx = int(fileString[12:18])
#         #print(int(fileString[12:18]))
#         darkFilesDict[idx] = fileString
#         darkFiles.append(fileString)


# next is to find the dark files for each file
darkFilesDict4Idx = {}

# park_ss_ff

for fileString in listFiles:
    if fileString.startswith('park_ss_ff'):
        # find the index of each file
        x = fileString.split("_")
        idx = int(x[-1].split(".")[0])
        # next is to find the corresponding dark file
        for darkKey in darkFilesKeys:
            if darkKey >= idx:
                print(str(idx) + "-" + str(darkKey))
                darkFilesDict4Idx[fileString] = darkFilesDict[darkKey]
                break


# now start to convert ge5 files to patches
from src.datasets.ge2patch import *


outDir = '../test_patches_v1/'

i = 1

for key in darkFilesDict4Idx:
    print(f"Reading dark file from {dataDir+darkFilesDict4Idx[key]} ... ")
    dark = ge_raw2array(dataDir+darkFilesDict4Idx[key], skip_frm=0).mean(axis=0).astype(np.float32)
    print(f"Done with reading dark file from {dataDir+darkFilesDict4Idx[key]} with {dark}")
    print(f"The input file is {key}")
    outFile = outDir + key[:-3] + "h5"
    print(f"The output file is {outFile}")
    print(f"Start to generate patches for file {key} ({i}/{len(darkFilesDict4Idx)})...")
    ge_raw2patch(gefname=dataDir+key, ofn=outFile, dark=dark, 
                 bkgd=100, psz=15, skip_frm=0, min_intensity=0, max_r=None)
    print(f"Done generating patches for file {key} ({i}/{len(darkFilesDict4Idx)})...")
    i += 1





# In[ ]:





# In[ ]:




