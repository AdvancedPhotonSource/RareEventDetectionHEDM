import numpy as np
from numpy import matlib
import os
import math
import random

import fabio
import warnings
import logging

# library for patch extraction
import cv2, os, h5py

def frame_peak_patches_cv2(frame, psz, angle, min_intensity=0, max_r=None):
    fh, fw = frame.shape
    patches, peak_ori = [], []
    mask = (frame > min_intensity).astype(np.uint8)
    comps, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    big_peaks = 0
    small_pixel_peak = 0
    for comp in range(1, comps):
        # ignore single-pixel peak
        if stats[comp, cv2.CC_STAT_WIDTH] < 3 or stats[comp, cv2.CC_STAT_HEIGHT] < 3: 
            small_pixel_peak += 1
            continue 
            
        # ignore component that is bigger than patch size
        if stats[comp, cv2.CC_STAT_WIDTH] > psz or stats[comp, cv2.CC_STAT_HEIGHT] > psz:
            big_peaks += 1
            continue
        
        # check if the component is within the max radius
        c, r = centroids[comp, 0], centroids[comp, 1]
        if max_r is not None and max_r**2 < ((c - fw/2)**2 + ( r - fh/2)**2):
            continue
                    
        col_s = stats[comp, cv2.CC_STAT_LEFT]
        col_e = col_s + stats[comp, cv2.CC_STAT_WIDTH]
        
        row_s = stats[comp, cv2.CC_STAT_TOP]
        row_e = row_s + stats[comp, cv2.CC_STAT_HEIGHT]

        _patch = frame[row_s:row_e, col_s:col_e]
        
        # mask out other labels in the patch
        _mask  = cc_labels[row_s:row_e, col_s:col_e] == comp
        _patch = _patch * _mask

        if _patch.size != psz * psz:
            h, w = _patch.shape
            _lp = (psz - w) // 2
            _rp = (psz - w) - _lp
            _tp = (psz - h) // 2
            _bp = (psz - h) - _tp
            _patch = np.pad(_patch, ((_tp, _bp), (_lp, _rp)), mode='constant', constant_values=0)
        else:
            _tp, _lp = 0, 0

        _min, _max = _patch.min(), _patch.max()
        if _min == _max: continue

        _pr_o = row_s - _tp
        _pc_o = col_s - _lp
        peak_ori.append((angle, _pr_o, _pc_o))
        patches.append(_patch)

    return np.array(patches).astype(np.float16), np.array(peak_ori), big_peaks

def ge_raw2array(gefname, skip_frm=0):
    det_res = 2048
    frame_sz= det_res * det_res * 2
    head_sz = 8192 + skip_frm * frame_sz # skip frames as needed
    n_frame = int((os.stat(gefname).st_size - head_sz) / frame_sz)
    mod = (os.stat(gefname).st_size - head_sz) % frame_sz
    if mod != 0:
        print("data in the file are not completely parsed, %d left over" % mod)
        
    with open(gefname, "rb") as fp:
        fp.seek(head_sz, os.SEEK_SET)
        frames = np.zeros((n_frame, det_res, det_res), dtype=np.uint16)
        for i in range(n_frame):
            frames[i] = np.fromfile(fp, dtype=np.uint16, count=det_res*det_res).reshape(det_res, det_res)
    return frames

def ge_raw2array_fabio(gefname, skip_frm=0):

    # add this line to suppress some warnings
    logging.getLogger("fabio").setLevel(logging.ERROR)

    # Load the image file
    image = fabio.open(gefname)

    # Check if the file supports multiple frames
    try:
        nframes = int(image.nframes)  # AttributeError if not supported
        print("Number of frames:", nframes)
    except AttributeError:
        nframes = 1
        print("Number of frames:", nframes)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)  # Replace with the relevant category
        frames = [image.get_frame(i).data for i in range(skip_frm, nframes)]

    # Convert the list of frames to a 3D NumPy array
    frames_array = np.array(frames)

    return frames_array

def ge_raw2patch(gefname, ofn, dark, bkgd, psz, skip_frm=0, min_intensity=0, max_r=None):
    frames = ge_raw2array_fabio(gefname, skip_frm=1)
    
    if not isinstance(dark, str):
        frames = frames.astype(np.float32) - dark

    if bkgd > 0:
        frames[frames < bkgd] = 0
    frames = frames.astype(np.uint16)
    
    patches, peak_ori = [], []
    frames_idx = []

    too_big_peaks = 0
    for i in range(frames.shape[0]):
        _pc, _ori, _bp = frame_peak_patches_cv2(frames[i], angle=i, psz=psz, min_intensity=0, max_r=None)
        if(_pc.size == 0):
            continue
        patches.append(_pc)
        peak_ori.append(_ori)
        frames_idx.append([i] * _pc.shape[0])
        too_big_peaks += _bp

    patches = np.concatenate(patches,  axis=0)
    peak_ori= np.concatenate(peak_ori, axis=0)
    frames_idx = np.concatenate(frames_idx, axis=0)

    print(f"{patches.shape[0]} patches of size {psz} cropped from {gefname}, {too_big_peaks} are too big.")
    with h5py.File(ofn, 'w') as h5fd:
        h5fd.create_dataset('patch', data=patches, dtype=np.uint16)
        h5fd.create_dataset('coordinate', data=peak_ori, dtype=np.uint16)
        h5fd.create_dataset('frame_idx', data=frames_idx, dtype=np.uint16)

# scan from wdir and return a list of datasets
def find_dataset_pooling(dataDir, bkgd, datasetPre):
    
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
def find_dataset_single(idata, idark, bkgd, datasetPre):

    print(f"Raw file based mode is enabled, now need to process {idata} and substract it by the dark file {idark} if provided")

    listFiles = list(idata)

    # need to add the function to subtract the dark file, patch extraction and output to a h5 file
    # read the raw scan and dark file and output a h5 file for later processing    
    if idark != "dark":
        print(f"Reading dark file from {idark} ... ")
        dark = ge_raw2array_fabio(idark, skip_frm=0).mean(axis=0).astype(np.float32)
        print(f"Done with reading dark file from {idark}")
    else:
        print(f"no dark file provided, skip dark file reading")


    outFile = f"{idata}.h5"
    print(f'the output h5 file is:{outFile}')

    print(f"Reading baseline/testing file from {idata} ... ")    
    if idark != "dark":
        ge_raw2patch(gefname=idata, ofn=outFile, dark=dark, bkgd=bkgd, psz=15, skip_frm=0, \
                    min_intensity=0, max_r=None)
    else:
        ge_raw2patch(gefname=idata, ofn=outFile, dark=idark, bkgd=bkgd, psz=15, skip_frm=0, \
                         min_intensity=0, max_r=None)
    print(f"Done with reading baseline/training file from {idata}")

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

    return [str(outFile)], filesPressure, filesIdx


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