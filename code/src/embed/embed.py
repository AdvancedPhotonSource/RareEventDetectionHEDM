
"""

The base Embed class.


"""

import h5py, torch
import numpy as np
import math
import random

from src.util.utility import * 

class Embed:

    def __init__(self, mdlfn):
        self._bragg_emb_model = torch.jit.load(mdlfn, map_location='cpu')

    def load_model(self, mdlfn):
        self._bragg_emb_model = torch.jit.load(mdlfn, map_location='cpu')
    
    def peak2emb_sparseview(self, patch_h5, frms=None):
        with h5py.File(patch_h5, 'r') as fp:
            coordinate = fp['coordinate'][:]
            patches    = fp['patch'][:].astype(np.float32)
            if frms is not None:
                frm_mask = np.array([False,]*(1+coordinate[:,0].max()))
                frm_mask[::frms] = True
                ptc_mask = np.array([frm_mask[i] for i in coordinate[:,0]])
                patches  = patches[ptc_mask]
            print(f"{patches.shape[0]} out of {coordinate.shape[0]} patches used, with frms={frms} sparse viewed")

        _min = patches.min(axis=(1, 2))[:,np.newaxis,np.newaxis]
        _max = patches.max(axis=(1, 2))[:,np.newaxis,np.newaxis]
        patches = ((patches - _min) / (_max - _min)).astype(np.float32)

        with torch.no_grad():
            embd = self._bragg_emb_model.forward(torch.from_numpy(patches[:,None])).numpy()
        return embd

    # generate the embedding 
    def peak2emb_missingwedge(self, patch_h5, frms=None, degree=360, degs_mode=1, seed=0):
        with h5py.File(patch_h5, 'r') as fp:
            coordinate = fp['coordinate'][:]
            patches    = fp['patch'][:].astype(np.float32)
        
            if degree != 360:
                # now need to select the patches that are in the degree range
                frame_idx = fp['frame_idx'][:]
                num_patches = coordinate.shape[0]
                # we need to add a check for whether we found dark spot
                counter = 0
                darkOrNot = 1
                while darkOrNot:  
                    patched_idx, darkOrNot = find_degree_pathches(num_patches, frame_idx, degree, degs_mode=degs_mode, seed=seed+counter)
                    if darkOrNot: print("find a dark dataset")
                    counter += 1
                # print(f"we spend {counter} iters to find the non dark partial dataset")

                patches = patches[patched_idx]
                coordinate = coordinate[patched_idx]

            # print(f"{patches.shape[0]} out of {total_patches} patches used, with first frms={frms}")


        _min = patches.min(axis=(1, 2))[:,np.newaxis,np.newaxis]
        _max = patches.max(axis=(1, 2))[:,np.newaxis,np.newaxis]
        patches = ((patches - _min) / (_max - _min)).astype(np.float32)

        #self._bragg_emb_model = torch.jit.load(mdlfn, map_location='cpu')

        with torch.no_grad():
            embd = self._bragg_emb_model.forward(torch.from_numpy(patches[:,None])).numpy()
        return embd, patches.shape[0] 

    

 

    
