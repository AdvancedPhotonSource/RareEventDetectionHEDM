from torch.utils.data import Dataset
import numpy as np
import h5py, sys, torchvision, torch

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
    def __init__(self, ifn, psz=-1, train=True, tv_split=1):
        self.transform = data_transforms(psz)
        with h5py.File(ifn, 'r') as h5:
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
