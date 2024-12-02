import torch, torchvision, copy, logging, argparse, os, sys, shutil
import numpy as np
from model import MLPhead, BraggPeakBYOL, targetNN

def loss_fn(x, y):
    x = torch.nn.functional.normalize(x, dim=-1, p=2)
    y = torch.nn.functional.normalize(y, dim=-1, p=2)
    
    return 2 - 2 * (x * y).sum(dim=-1)

    
def data_transforms():
    blur = torchvision.transforms.GaussianBlur((3, 3), (1.0, 2.0))
    # get a set of data augmentation transformations 
    data_transforms = torchvision.transforms.Compose([#torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                        #   torchvision.transforms.RandomVerticalFlip(p=0.5),
                                          # torchvision.transforms.RandomErasing(p=0.2),
                                          torchvision.transforms.RandomRotation(degrees=30),
                                          torchvision.transforms.RandomApply([blur, ], p=.3)])
    return data_transforms


def main(args):
    torch_devs = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_ds = torchvision.datasets.MNIST('dataset/', train=True, download=True,
                 transform=torchvision.transforms.Compose([
                   torchvision.transforms.ToTensor(),]) )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.mbsz, shuffle=True)
    
    zdim = 96
    model = BraggPeakBYOL(psz=28, hdim=64, proj_dim=zdim)
    predictor = MLPhead(zdim, zdim, zdim)

    if torch.cuda.is_available():
        model = model.to(torch_devs)
        predictor = predictor.to(torch_devs)

    target_maker = targetNN(beta=0.996)
    target_maker.update(model)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr) 
    # optimizer = torch.optim.SGD(list(model.parameters()) + list(predictor.parameters()),  lr=args.lr, momentum=0.9, weight_decay=0.0004)

    logging.info("start training on %d samples with mbsz of %d" % (len(train_ds), args.mbsz))
    aug_trans = data_transforms()
    for ep in range(args.maxep):
        for i, (X, _) in enumerate(train_loader):
            X  = X.to(torch_devs)
            v1 = aug_trans(X)
            v2 = aug_trans(X)

            optimizer.zero_grad()
            
            online_p1 = predictor.forward(model.forward(v1, rety=False))
            online_p2 = predictor.forward(model.forward(v2, rety=False))

            target_p1    = target_maker.predict(v1)
            target_p2    = target_maker.predict(v2)
            
            loss = loss_fn(online_p1, target_p2) + loss_fn(online_p2, target_p1)
            loss = loss.mean()
            loss.backward()
            optimizer.step() 

            target_maker.update(model)

        logging.info("epoch %3d completed, training loss: %f" % (ep, loss.detach().cpu().numpy()))
        torch.save(model.state_dict(), "%s-itrOut/mdl-it%05d.pth" % (args.expName, ep, ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bragg peak finding for HEDM.')
    parser.add_argument('-gpus',   type=str, default="0", help='list of visiable GPUs')
    parser.add_argument('-expName',type=str, default="debug", help='Experiment name')
    parser.add_argument('-lr',     type=float,default=3e-4, help='learning rate')
    parser.add_argument('-mbsz',   type=int, default=128, help='mini batch size')
    parser.add_argument('-maxep',  type=int, default=100, help='max training epoches')

    args, unparsed = parser.parse_known_args()

    itr_out_dir = args.expName + '-itrOut'
    if os.path.isdir(itr_out_dir): 
        shutil.rmtree(itr_out_dir)
    os.mkdir(itr_out_dir) # to save temp output

    logging.basicConfig(filename=os.path.join(itr_out_dir, 'BYOL.log'), level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    main(args)
    