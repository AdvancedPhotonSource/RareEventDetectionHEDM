"""
read data from h5 files, find the peak,
return a list of patches in a h5 file

"""

import argparse, os, time, h5py, glob, cv2
import numpy as np
import pandas as pd 

import fabio
import warnings

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


# input: a list of h5 files (multiple datasets)
# output: a concatenated h5 files (a single dataset)   
def concatenate_patches(path, list_patches, new_name):
    numDatasets = len(list_patches)
    print(f"{numDatasets} datasets will be concatenated")
    
    resultPatchArray = h5py.File(path+list_patches[0],'a')['patch']
    resultCoArray = h5py.File(path+list_patches[0],'a')['coordinate']

    for i in range(1, numDatasets):
        fileTmp = h5py.File(path+list_patches[i],'a')
        print(f"concatenated {i+1} datasets")
        resultPatchArray = np.concatenate((resultPatchArray, fileTmp['patch']), axis=0)
        resultCoArray = np.concatenate((resultCoArray, fileTmp['coordinate']), axis=0)
        fileTmp.close()

    with h5py.File(new_name, 'w') as h5fd:
        h5fd.create_dataset('patch', data=resultPatchArray, dtype=np.uint16)
        h5fd.create_dataset('coordinate', data=resultCoArray, dtype=np.uint16)

    return resultPatchArray




