
# this code is used to convert ge5 files to to hd5 files with peak info.
# input: the folder that contains the 
# the majority of this code is from ge2patch.py in v0

import argparse, os, time, h5py, glob, cv2
import numpy as np
import pandas as pd 
import os

from src.datasets.ge2patch import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bragg peak Analysis.')
    parser.add_argument('-ifd',   type=str, required=True, help='raw ge files directory')
    parser.add_argument('-dark',  type=str, required=True, help='dark ge file')
    parser.add_argument('-th',    type=int, default=100, help='threshold')
    parser.add_argument('-ofd',   type=str, required=True, help='processed h5 directory')
    parser.add_argument('-psz',   type=int, default=15, help='patch size')

    args, unparsed = parser.parse_known_args()

    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)
    
    
    dataDir = args.ifd
    #dataDir = '/local/s1iduser/hurley_feb22/ge5/' 
    
    listFiles = os.listdir(dataDir)
    
    geFilesString   = []
    geFilesPressure = []
    geFilesIdx      = []

    numDatasets = 0 
    for fileString in listFiles:
        if fileString.startswith('hurley_quartz'):
            geFilesString.append(fileString)
            pressure = int(fileString[14:16])
            geFilesPressure.append(pressure)
            idx = int(fileString[23:29])
            geFilesIdx.append(idx)
            numDatasets += 1

    print(f"There are {numDatasets} datasets in total")

    print(f"Reading dark file from {args.dark} ... ")
    dark = ge_raw2array(args.dark, skip_frm=0).mean(axis=0).astype(np.float32)
    print(f"Done with reading dark file from {args.dark}")

    for i in range(46, numDatasets):
        inFile = args.ifd + geFilesString[i]
        print(f"The input file is {inFile}")
        outFile = args.ofd + geFilesString[i][:-3] + "h5"
        print(f"The output file is {outFile}")
        print(f"Start to generate patches for file {geFilesString[i]} ({i}/{numDatasets})...")
        ge_raw2patch(gefname=inFile, ofn=outFile, dark=dark, bkgd=args.th, psz=15, skip_frm=0, min_intensity=0, max_r=None)
        print(f"Done generating patches for file {geFilesString[i]} ({i}/{numDatasets})...")

    
    print("Done")
