from scipy.spatial.distance import cosine, euclidean
import torch, h5py, torchvision
from random import sample
import numpy as np 

def load_patch_from_h5(ih5, norm=True):
    with h5py.File(ih5, 'r') as h5fp:
        patches = h5fp['patch'][:]
    if norm:
        _min = patches.min(axis=(1, 2))[:,np.newaxis,np.newaxis]
        _max = patches.max(axis=(1, 2))[:,np.newaxis,np.newaxis]
        patches = ((patches - _min) / (_max - _min)).astype(np.float32)[:,np.newaxis]
    return patches

def data_transforms(psz):
    # get a set of data augmentation transformations
    data_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                          torchvision.transforms.RandomVerticalFlip(p=0.5),
                                          torchvision.transforms.RandomErasing(p=0.2, scale=(1/psz, 4/psz), ratio=(0.5, 2)),
                                          torchvision.transforms.RandomRotation(degrees=180)])
    return data_transforms

def make_aug_views(img, n, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    transform = data_transforms(img.shape[-1])
    res = [img]
    img_torch = torch.from_numpy(img)
    for i in range(n):
        res.append(transform(img_torch))
    return np.concatenate(res, axis=0)

def emd_mdl_eva(ih5, mdlfn, rep=100, mb=31):
    emb_mdl = torch.jit.load(mdlfn, map_location='cpu')
    for key, p in emb_mdl.named_parameters(): p.requires_grad = False
    patches = load_patch_from_h5(ih5)
    distance_augs = []
    distance_rand = []
    for i in sample(range(0, patches.shape[0]), rep):
        aug_views = make_aug_views(patches[i], mb, seed=None) # will include ref as the 1st item in return
        embds = emb_mdl.forward(torch.from_numpy(aug_views[:,None]))
        distance_augs += [cosine(embds[0], _emb) for _emb in embds[1:]]
        
        batch_samples  = sample(range(0, patches.shape[0]), mb)
        rnd_samples    = np.concatenate([patches[k][None] for k in batch_samples])
        embds = emb_mdl.forward(torch.from_numpy(np.concatenate([patches[i][None], rnd_samples], axis=0)))
        distance_rand += [cosine(embds[0], _emb) for _emb in embds[1:]]
        
    return np.percentile(distance_rand, (50, 75, 95, 99)) / np.percentile(distance_augs, (50, 75, 95, 99))

for ep in range(1, 31):
    mdlfn= '/lambda_stor/data/zliu/BraggDP/BraggEmb/feb21-itrOut/script-ep%05d.pth' % ep
    rel  = emd_mdl_eva(ih5='/lambda_stor/data/zliu/DP-data/mar22-psz15/patch-525.h5', mdlfn=mdlfn, rep=100)
    print(ep, rel)