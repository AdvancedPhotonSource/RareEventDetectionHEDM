#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

dataDir = '/home/beams/S1IDUSER/mnt/orthros/PUP_AFRL_Dec14_data/GE'

listFiles = os.listdir(dataDir)
listFiles


# In[ ]:


ti7FilesList = []
ti7FilesDict4Idx = {}

for fileString in listFiles:
    if fileString.startswith('Ti7_23_'):
        # find the index of each file
        x = fileString.split("_")
        idx = int(x[-1].split(".")[0])
        ti7FilesDict4Idx[fileString] = idx
#         idx = int(fileString[12:18])
#         #print(int(fileString[12:18]))
#         darkFilesDict[idx] = fileString
#         darkFiles.append(fileString)

# 
ti7FilesDict4Idx 


# In[ ]:


# now start to convert ge5 files to patches
from src.datasets.ge2patch import *


# In[ ]:


darkFile = '/home/beams/S1IDUSER/mnt/orthros/PUP_AFRL_Dec14_data/GE/dark_0pt3s_00025.ge3'

print(f"Reading dark file from {darkFile} ... ")
dark = ge_raw2array(darkFile, skip_frm=0).mean(axis=0).astype(np.float32)


# In[ ]:


outDir = '/net/wolf/data/tmp_data_ti7_14/'

i = 1

for key in ti7FilesDict4Idx:
    if ti7FilesDict4Idx[key] >= 151 and ti7FilesDict4Idx[key] <= 758:
        print(f"The input file is {key}")
        outFile = outDir + key[:-3] + "h5"
        print(f"The output file is {outFile}")
        print(f"Start to generate patches for file {key} ({i}/{len(ti7FilesDict4Idx)})...")
        ge_raw2patch(gefname=dataDir+'/'+key, ofn=outFile, dark=dark, 
                     bkgd=100, psz=15, skip_frm=0, min_intensity=0, max_r=None)
        print(f"Done generating patches for file {key} ({i}/{len(ti7FilesDict4Idx)})...")
        i += 1


# In[ ]:





# In[ ]:




