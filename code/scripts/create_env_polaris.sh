#!/bin/bash -l

module load conda

conda create -n event_detection python=3.9

conda activate event_detection

pip install torch
pip install numpy
pip install h5py
pip install torchvision