from torch.utils.data import Dataset
import numpy as np
import h5py, sys, torchvision, torch

# library for patch extraction
import cv2, os
import fabio
import warnings
import logging

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

    # # Display shape of the resulting array
    # print("Shape of frames array:", frames_array.shape)

    # # Optionally, display metadata for the first skipped frame (0th frame)
    # if nframes > 0:
    #     print("Metadata of skipped frame 0:", image.get_frame(0).header)

    return frames_array

def ge_raw2patch(gefname, ofn, dark, thold, psz, skip_frm=0, min_intensity=0, max_r=None):
    # frames = ge_raw2array(gefname, skip_frm=1)
    frames = ge_raw2array_fabio(gefname, skip_frm=1)

    if not isinstance(dark, str):
        frames = frames.astype(np.float32) - dark
    
    if thold > 0:
        frames[frames < thold] = 0
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

def data_transforms(psz):
    # get a set of data augmentation transformations 
    data_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                          torchvision.transforms.RandomVerticalFlip(p=0.5),
                                          torchvision.transforms.RandomErasing(p=0.2, scale=(1/psz, 4/psz), ratio=(0.5, 2)),
                                          torchvision.transforms.RandomRotation(degrees=180)])
    return data_transforms

class BraggDatasetMIDAS(Dataset):
    def __init__(self, ifn, psz=-1, train=True, tv_split=1):
        self.transform = data_transforms(psz)
        with h5py.File(ifn, 'r') as h5:
            train_N = int(tv_split * h5['patch'].shape[0])
            if train:
                sidx, eidx = 0, train_N
            else:
                sidx, eidx = train_N, None
            patches = h5['patch'][sidx:eidx]
            peakLoc = h5['peakLoc'][sidx:eidx]

        nPeaks = np.array([_pl.shape[0]//2 for _pl in peakLoc])
        sel_peak_patches = patches[nPeaks >= 1]
        _min = sel_peak_patches.min(axis=(1, 2))[:,np.newaxis,np.newaxis]
        _max = sel_peak_patches.max(axis=(1, 2))[:,np.newaxis,np.newaxis] + 1e-10
        self.patches = ((sel_peak_patches - _min) / (_max - _min)).astype(np.float32)[:,np.newaxis]

        self.fpsz    = self.patches.shape[-1]

        self.psz = self.fpsz if psz <= 0 else psz

        if self.psz > self.fpsz:
            sys.exit(f"It's impossible to make patch with ({self.psz}, {self.psz}) from ({self.fpsz}, {self.fpsz})")

    def __getitem__(self, idx):
        sr  = np.random.randint(0, self.fpsz-self.psz+1)
        sc  = np.random.randint(0, self.fpsz-self.psz+1)
        c_patch = self.patches[idx, :, sr:(sr+self.psz), sc:(sc+self.psz)]

        c_patch = torch.from_numpy(c_patch)
        view1 = c_patch
        view2 = self.transform(c_patch)

        return view1, view2

    def __len__(self):
        return self.patches.shape[0]

class BraggDataset(Dataset):
    def __init__(self, irawt, irawd, thold, psz=-1, train=True, tv_split=1):
        self.transform = data_transforms(psz)

        # read the raw scan and dark file and output a h5 file for later processing
        if irawd != "default_dark":
            print(f"Reading dark file from {irawd} ... ")
            dark = ge_raw2array_fabio(irawd, skip_frm=0).mean(axis=0).astype(np.float32)
            print(f"Done with reading dark file from {irawd}")
        else:
            print(f"no dark file provided, skip dark file reading")

        outFile = "test.h5"
        print(f"Reading training file from {irawt} ... ")
        if irawd != "default_dark":
            ge_raw2patch(gefname=irawt, ofn=outFile, dark=dark, thold=thold, psz=15, skip_frm=0, \
                         min_intensity=0, max_r=None)
        else:
            ge_raw2patch(gefname=irawt, ofn=outFile, dark=irawd, thold=thold, psz=15, skip_frm=0, \
                         min_intensity=0, max_r=None)
        print(f"Done with reading training file from {irawt}")

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

    def __getitem__(self, idx):
        sr  = np.random.randint(0, self.fpsz-self.psz+1)
        sc  = np.random.randint(0, self.fpsz-self.psz+1)
        c_patch = self.patches[idx, :, sr:(sr+self.psz), sc:(sc+self.psz)]

        c_patch = torch.from_numpy(c_patch)
        view1 = c_patch
        view2 = self.transform(c_patch)

        return view1, view2

    def __len__(self):
        return self.patches.shape[0]
