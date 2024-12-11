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
    parser = argparse.ArgumentParser(description='run anaomaly detection')

    # Input and output
    parser.add_argument('-embmdl',   type=str, required=True, help='embedding model path')
    parser.add_argument('-streaming_mode',type=int, default="1", help='streaming mode (1) or not (0), for streaming mode (1), the baseline, testing, and dark file will be raw files')

    parser.add_argument('-ibase',    type=str, required=True, help='baseline dataset, it must be a raw file if streaming mode is selected')
    parser.add_argument('-itest',    type=str, required=True, help='testing scan(s), it is a folder of all scans if pooling mode is selected, it is a single file if streaming mode is selected')

    parser.add_argument('-ocsv',     type=str, required=True, help='result csv file')

    parser.add_argument('-frms',     type=int, default=None, help='frames subsampling')
    parser.add_argument('-dpre',     type=str, default="park_ss_ff", help='dataset file prefix, required for pooling mode, e.g., hurley_ff...')

    # hyper parameters tuned
    parser.add_argument('-uqthr',    type=float, default=0.4, help='threshold for confidence for UQ')
    parser.add_argument('-ncluster', type=int, default=30, help='number of clusters')
    parser.add_argument('-cluster',  type=str, default="Kmeans", help='type of clustering algorithm')

    # partial test dataset related arguments
    parser.add_argument('-degs',      type=int, default="360", help='number of degree range for testing, when it is 360, it means full range')
    parser.add_argument('-seed',      type=int, default="0",   help='random seed for testing')
    parser.add_argument('-degs_mode', type=int, default="1",   help='mode of the degree test: 1: normal mode, 0: debug mode: sampling 20 times')

    # argument for streaming version of code
    parser.add_argument('-idarkbase', type=str, default="dark", help='input dark file for the baseline scan, it will be required if the streaming version is enabled')
    parser.add_argument('-idarktest', type=str, default="dark", help='input dark file for the testing scan, it will be required if the streaming version is enabled')


    args, unparsed = parser.parse_known_args()

    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)

    Detection_run = Run(args)

    Detection_run.start()
