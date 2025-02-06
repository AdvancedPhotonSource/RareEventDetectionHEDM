'''
input:
- a trained embedding model
- a dataset as baseline
- a folder of dataset to evaluate

process:
- run inference on the baseline model
- then train clustering model, e.g., GMM

- inference on all datasets, one by one
- inference on clustering model to get both cluster distribution and UQ
'''

import argparse
import os

from src.system_run.run_detection import DetectionRun as Run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run anomaly detection')

    # Input and output
    parser.add_argument('-file_mode', type=int, default="0", help='file-based mode or not, if enabled, the input needs to be a raw ge5 file')
    parser.add_argument('-patch_mode',type=int, default="0", help='patch mode or not, if enabled, the input needs to be a h5 file which has all patches')
    parser.add_argument('-thold',      type=int, required=True, help='background value to process raw images')

    parser.add_argument('-trained_encoder', type=str, default="../BraggEmb_code/model_save-itrOut/script-ep00100.pth", help='trained encoder model from the prevoius step embedding model path')
    
    parser.add_argument('-baseline_scan', type=str, required=True, help='baseline dataset, it must be a raw file if file-based mode is selected')

    parser.add_argument('-frms',     type=int, default=None, help='frames subsampling')
    parser.add_argument('-dpre',     type=str, default="park_ss_ff", help='dataset file prefix, required for patch mode, e.g., hurley_ff...')

    # hyper parameters tuned
    parser.add_argument('-ncluster', type=int, default=40, help='number of clusters')
    parser.add_argument('-uqthr',    type=float, default=0.4, help='threshold for confidence for UQ')
    parser.add_argument('-cluster',  type=str, default="Kmeans", help='type of clustering algorithm')

    # partial test dataset related arguments
    parser.add_argument('-degs',      type=int, default="360", help='number of degree range for testing, when it is 360, it means full range')
    parser.add_argument('-seed',      type=int, default="0",   help='random seed for testing')
    parser.add_argument('-degs_mode', type=int, default="1",   help='mode of the degree test: 1: normal mode, 0: debug mode: sampling 20 times')

    # argument for streaming version of code
    parser.add_argument('-baseline_scan_dark', type=str, default="dark", help='input dark file for the baseline scan, it will be required if the streaming version is enabled')

    args, unparsed = parser.parse_known_args()

    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)

    Baseline_run = Run(args)

    Baseline_run.baseline_process()
