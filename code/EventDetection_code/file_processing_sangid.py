#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

dataDir = '/home/beams/S1IDUSER/mnt/s1c/sangid_aug19_data/sangid_aug19/ge3/'

listFiles = os.listdir(dataDir)
listFiles


# In[ ]:


darkFilesList = []
darkFilesDict = {}

for fileString in listFiles:
    if fileString.startswith('dark_before'):
        # find the index of each dark before
        idx = int(fileString[12:18])
        #print(int(fileString[12:18]))
        darkFilesDict[idx] = fileString

darkFilesDict


# In[ ]:


darkFilesDict


# In[ ]:


darkFilesKeys = sorted(darkFilesDict)
darkFilesKeys


# In[ ]:


parkFilesList = []
parkFilesDict4Idx = {}

# park_ss_ff

for fileString in listFiles:
    if fileString.startswith('lshr'):
        # find the index of each file
        x = fileString.split("_")
        idx = int(x[-1].split(".")[0])
        parkFilesDict4Idx[fileString] = idx
#         idx = int(fileString[12:18])
#         #print(int(fileString[12:18]))
#         darkFilesDict[idx] = fileString
#         darkFiles.append(fileString)

# 
parkFilesDict4Idx 


# In[ ]:


# next is to find the dark files for each file
darkFilesDict4Idx = {}

# park_ss_ff

for fileString in listFiles:
    if fileString.startswith('lshr_'):
        # find the index of each file
        x = fileString.split("_")
        idx = int(x[-1].split(".")[0])
        # next is to find the corresponding dark file
        for darkKey in darkFilesKeys:
            if darkKey >= idx:
                print(str(idx) + "-" + str(darkKey))
                darkFilesDict4Idx[fileString] = darkFilesDict[darkKey]
                break

darkFilesDict4Idx


# In[ ]:


len(darkFilesDict4Idx)


# In[ ]:


# now start to convert ge5 files to patches
from src.datasets.ge2patch import *


# In[ ]:


outDir = '../sangid_patch_dir/'

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





# In[ ]:





# In[ ]:




