from model import model_init, BraggPeakBYOL, targetNN, MLPhead
from loss  import regression_loss
import torch, argparse, os, time, sys, shutil, logging
from util import s2ituple, str2bool
from torch.utils.data import DataLoader
from dataset import BraggDataset
import numpy as np
from time import perf_counter

parser = argparse.ArgumentParser(description='Bragg peak finding for HEDM.')
parser.add_argument('-gpus',   type=str, default="0", help='the GPU to use')
parser.add_argument('-expName',type=str, default="model_save", help='Experiment name')
parser.add_argument('-lr',     type=float,default=3e-4, help='learning rate')
parser.add_argument('-mbsz',   type=int, default=128, help='mini batch size')
parser.add_argument('-maxep',  type=int, default=100, help='max training epoches')

parser.add_argument('-irawt',  required=True, help='input train raw scan file')
parser.add_argument('-irawd',  required=True, help='input dark raw scan file')

parser.add_argument('-verbose',type=int, default=1, help='non-zero to print logs to stdout')
parser.add_argument('-psz',    type=int, default=15, help='training/model patch size')
parser.add_argument('-zdim',   type=int, default=64, help='projection(z)/prediction dim')
parser.add_argument('-nworks', type=int, default=4, help='number of workers for data loading')

args, unparsed = parser.parse_known_args()

if len(unparsed) > 0:
    print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
    exit(0)

def main(args):
    
    total_time = 0
    total_comp_time = 0
    total_2dev_time = 0
    total_cpul_time = 0
    total_logg_time = 0

    data_obtain_time1 = 0
    data_obtain_time2 = 0

    total_time_tick = perf_counter()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    os.environ["WORLD_SIZE"] = "1"

    torch_devs = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("[Info] loading data into CPU memory, it will take a while ... ...")

    print(torch_devs)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    time_datl_tick = perf_counter()

    # modified data loader here to read the raw file 
    train_ds = BraggDataset(args.irawt, args.irawd, psz=args.psz, train=True) 

    print(f'data load phase 1 time is {perf_counter()-time_datl_tick}')

    time_datl_tick = perf_counter()

    train_dl = DataLoader(dataset=train_ds, batch_size=args.mbsz, shuffle=True,\
                   num_workers=args.nworks, prefetch_factor=args.mbsz, drop_last=True, pin_memory=True)
    logging.info("%d samples loaded for training" % len(train_ds))

    print(f'data load phase 2 time is {perf_counter()-time_datl_tick}')
    
    time_mdli_tick = perf_counter()

    model = BraggPeakBYOL(psz=args.psz, hdim=64, proj_dim=args.zdim)
    predictor = MLPhead(args.zdim, args.zdim, args.zdim)

    print(f'model initial phase 1 time is {perf_counter()-time_mdli_tick}')

    time_mdli_tick = perf_counter()

    if torch.cuda.is_available():
        gpus = torch.cuda.device_count()
        print(gpus)
        if gpus > 1:
            logging.info("This implementation only makes use of one GPU although %d are visiable" % gpus)
        model = model.to(torch_devs)
        predictor = predictor.to(torch_devs)
    
    print(f'model initial phase 2 time is {perf_counter()-time_mdli_tick}')

    time_mdli_tick = perf_counter()

    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr) 

    target_maker = targetNN(beta=0.996)
    target_maker.update(model)

    print(f'model initial phase 3 time is {perf_counter()-time_mdli_tick}')

    for epoch in range(args.maxep):
        ep_tick = perf_counter()
        time_comp = 0
        prev_iter_end_time = None

        for v1, v2 in train_dl:
            time_2dev_tick = perf_counter()

            if prev_iter_end_time is not None:
                total_cpul_time += time_2dev_tick - prev_iter_end_time
            else:
                total_cpul_time += time_2dev_tick - ep_tick

            start.record()
            v1 = v1.to(torch_devs)
            v2 = v2.to(torch_devs)
            
            end.record()
            torch.cuda.synchronize()
            total_2dev_time += start.elapsed_time(end)

            # print(start.elapsed_time(end))
            # total_2dev_time += start.elapsed_time(end)
            # total_2dev_time += perf_counter() - time_2dev_tick

            it_comp_tick = perf_counter()
            start.record()

            optimizer.zero_grad()

            y_rep_v1, z_proj_v1 = model.forward(v1, rety=False)
            y_rep_v2, z_proj_v2 = model.forward(v2, rety=False)

            online_p1 = predictor.forward(z_proj_v1)
            online_p2 = predictor.forward(z_proj_v2)

            target_p1 = target_maker.predict(v1)
            target_p2 = target_maker.predict(v2)
            
            loss = regression_loss(online_p1, target_p2) + regression_loss(online_p2, target_p1) #+\
                #    y_rep_v1.abs().sum(axis=-1) + y_rep_v2.abs().sum(axis=-1)
            loss = loss.mean()
            loss.backward()
            optimizer.step() 

            target_maker.update(model)

            time_comp += 1000 * (perf_counter() - it_comp_tick)
            end.record()
            torch.cuda.synchronize()
            total_comp_time += start.elapsed_time(end)

            prev_iter_end_time = perf_counter()

        time_e2e = 1000 * (perf_counter() - ep_tick)
        _prints = '[Info] @ %.1f Epoch: %05d, loss: %.4f, elapse: %.2fms/epoch (computation=%.1fms/epoch, %.2f%%)' % (\
                   perf_counter(), epoch, loss.cpu().detach().numpy(), time_e2e, time_comp, 100*time_comp/time_e2e)
        logging.info(_prints)

        start.record()
        torch.jit.save(torch.jit.trace(model, v1), "%s/script-ep%05d.pth" % (itr_out_dir, epoch+1))
        end.record()
        torch.cuda.synchronize()
        total_logg_time += start.elapsed_time(end)

    total_time = perf_counter() - total_time_tick
    print(f"total 2 dev time is {total_2dev_time/1000.0}, \ntotal real compute time is {total_comp_time/1000.0}, \
    \ntotal model saving time is {total_logg_time/1000.0}, \ntotal loading from cpu time is {total_cpul_time}, \
    \nthe total time is {total_time} s")

if __name__ == "__main__":
    if len(args.gpus) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    itr_out_dir = args.expName + '-itrOut'
    if os.path.isdir(itr_out_dir): 
        shutil.rmtree(itr_out_dir)
    os.mkdir(itr_out_dir) # to save temp output

    logging.basicConfig(filename=os.path.join(itr_out_dir, '%s.log' % args.expName), level=logging.DEBUG)
    if args.verbose != 0:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    main(args)
