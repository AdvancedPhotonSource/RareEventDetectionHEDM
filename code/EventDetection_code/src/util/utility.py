import numpy as np
from numpy import matlib
import os
import math
import random

# scan from wdir and return a list of datasets
def find_dataset_pooling(dataDir, datasetPre):
    
    listFiles = os.listdir(dataDir)
    
    filesString   = []
    filesPressure = []
    filesIdx      = []

    numDatasets = 0 
    for fileString in listFiles:
        if fileString.startswith(datasetPre):
            print(fileString)
            filesString.append(fileString)
            #pressure = int(fileString[14:16])
            #filesPressure.append(pressure)
            x = fileString.split("_")
            idx = int(x[-1].split(".")[0])
            #idx = int(fileString[23:29])
            filesIdx.append(idx)
            numDatasets += 1

    print(f"There are {numDatasets} patch datasets in total")
    #print(filesString)
    #print(filesPressure)
    #print(filesIdx) 

    return filesString, filesPressure, filesIdx


# finish a function here to 
def find_datase_single(idata, idark):

    print(f"Streaming mode is enabled, now need to process {idata} and sbstract it by the dark file {idark}")

    listFiles = list(idata)

    # need to add the function to subtract the dark file, patch extraction and output to a h5 file
    # read the raw scan and dark file and output a h5 file for later processing
    print(f"Reading dark file from {idark} ... ")
    dark = ge_raw2array(idark, skip_frm=0).mean(axis=0).astype(np.float32)
    print(f"Done with reading dark file from {irawd}")


    outFile = f"{idata}.h5"
    print(f"Reading baseline/testing file from {idata} ... ")
    ge_raw2patch(gefname=idata, ofn=outFile, dark=dark, bkgd=100, psz=15, skip_frm=0, \
                    min_intensity=0, max_r=None)
    print(f"Done with reading baseline/training file from {idata}")

    with h5py.File(outFile, 'r') as h5:
        train_N = int(tv_split * h5['patch'].shape[0])
        if train:
            sidx, eidx = 0, train_N
        else:
            sidx, eidx = train_N, None
        patches = h5['patch'][sidx:eidx]

    _min = patches.min(axis=(1, 2))[:,np.newaxis,np.newaxis]
    _max = patches.max(axis=(1, 2))[:,np.newaxis,np.newaxis]
    self.patches = ((patches - _min) / (_max - _min)).astype(np.float32)[:,np.newaxis]

    self.fpsz    = self.patches.shape[-1]

    self.psz = self.fpsz if psz <= 0 else psz

    if self.psz > self.fpsz:
        sys.exit(f"It's impossible to make patch with ({self.psz}, {self.psz}) from ({self.fpsz}, {self.fpsz})")

    filesString   = []
    filesPressure = []
    filesIdx      = []

    numDatasets = 0 
    for fileString in listFiles:
        if fileString.startswith(datasetPre):
            print(fileString)
            filesString.append(fileString)
            #pressure = int(fileString[14:16])
            #filesPressure.append(pressure)
            x = fileString.split("_")
            idx = int(x[-1].split(".")[0])
            #idx = int(fileString[23:29])
            filesIdx.append(idx)
            numDatasets += 1

    print(f"There are {numDatasets} patch datasets in total")
    #print(filesString)
    #print(filesPressure)
    #print(filesIdx) 

    return list(outFile), filesPressure, filesIdx


# find a list of patches from a dataset based on the degree range
# inputs: frame_idx: a list of frame index
#         degree: the degree range
#         seed: random seed to control the starting degree
# outputs: a list of patch index

def find_degree_pathches(num_patches, frame_idx, degree, seed=0, degs_mode=1):

    # first is to find the frame index range
    frameKeys = []
    frameDict = {}
    patchIdx = 0
    for frameIdx in frame_idx:
        frameIdx = int(frameIdx)
        if frameIdx not in frameKeys:
            frameKeys.append(frameIdx)
            frameDict[frameIdx] = []
            frameDict[frameIdx].append(patchIdx)
        else:
            frameDict[frameIdx].append(patchIdx)
        patchIdx += 1
    
    # find the average number of frames per degree
    numFrames = max(frameKeys) + 1
    avenumFramesperDegree = math.ceil(numFrames/360)
    avenumPatchesperDegree = math.ceil(num_patches/360)

    random.seed(seed)
    startDeg = random.randrange(0, 360)

    selectDeg = startDeg
    patchIdxs = []

    startFrame = selectDeg * avenumFramesperDegree
    while startFrame not in frameKeys:
        startFrame += 1
        # the following code is used to avoid overflow for starting degree
        if startFrame > max(frameKeys):
            startFrame = 0
    frameIdx = frameKeys.index(startFrame)
    frame = frameKeys[frameIdx]

    frameIdxSteps = 0
    frameSteps = 0

    # use a flag here to indicate whether we encounterd some dark field
    # if deg_mode == 0, we will discard those sampling results with dark field
    darkOrNot = 0
    count     = 0
    while len(patchIdxs) <= avenumPatchesperDegree*degree:
        # extract all patches in this frame
        patchIdxs += frameDict[frame]
        
        frameIdx += 1
        frameIdxSteps += 1

        # dark field should not happend at start
        if frameIdx >= len(frameKeys): 
            frameIdx -= len(frameKeys)
            frame = frameKeys[frameIdx]
            frameIdxSteps = 0 
            frameSteps    = 0
        elif count == 0:
            frame = frameKeys[frameIdx]              
            frameSteps += 1
        else: 
            oldFrame = frame
            frame = frameKeys[frameIdx]              
            frameSteps += frame
            frameSteps -= oldFrame 

        count += 1

        if frameSteps != frameIdxSteps:
            darkOrNot = 1
            # print(f"a dark file is found at frame {frame} with start {startFrame} and {avenumFramesperDegree*5}")

    return patchIdxs, darkOrNot