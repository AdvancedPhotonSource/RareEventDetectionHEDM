# RareEventDetectionHEDM
Rapid detection of rare events from in situ X-ray diffraction data using machine learning

<p float="left">
  <img src="doc/REI_schematic.png" width="250" />
  <img src="/doc/REI-detailed-schematic.png" width="250" /> 
</p>

## Installation

Weijian: please change this!!! this is just an example from https://github.com/saugatkandel/fast_smart_scanning

We recommend creating a new environment (through conda, pipenv, etc) to try out RareEventDetectionHEDM. The conda command might look like:
```shell
conda create -n foo_test python=3.10
conda activate foo_test
conda config --prepend channels conda-forge
conda install pip
conda install fabio ???
```
We recommend using Python 3.10? One alternative is to install scikit-image through conda instead. We have tested in this RHEL 8, but we do not expect any issues. Need a GPU? etc ...

## Usage

This code has been mostly tested with GE detector data at 1-ID. 

First, train the BYOL encoder on a baseline dataset (e.g., zero load):
```shell
python train_REI_encoder.py --input scan_001_0MPa.ge5 --output scan_001_0MPa.encoder
```
Second, calculate REI values for subsequent datasets (i.e., scans at different loads):
```shell
python calculate_REI.py --input scan_001_100MPa.ge5
python calculate_REI.py --input scan_001_110MPa.ge5
...
```


## Citation
If you use this code for your research, please cite our paper(s):
- W. Zheng, J.-S. Park, P. Kenesei, A. Ali, Z. Liu, I. Foster, N. Schwarz, R. Kettimuthu,
A. Miceli, and H. Sharma, “Rapid detection of rare events from in situ X-ray diffraction data
using machine learning,” Journal of Applied Crystallography, vol. 57, Aug 2024. (http://dx.doi.org/10.1107/S160057672400517X)

Or via bibtex

```
@article{Zheng:nb5377,
author = "Zheng, Weijian and Park, Jun-Sang and Kenesei, Peter and Ali, Ahsan and Liu, Zhengchun and Foster, Ian and Schwarz, Nicholas and Kettimuthu, Rajkumar and Miceli, Antonino and Sharma, Hemant",
title = "{Rapid detection of rare events from {\it in situ} X-ray diffraction data using machine learning}",
journal = "Journal of Applied Crystallography",
year = "2024",
volume = "57",
number = "4",
pages = "",
month = "Aug",
doi = {10.1107/S160057672400517X},
url = {https://doi.org/10.1107/S160057672400517X},
abstract = {High-energy X-ray diffraction methods can non-destructively map the 3D microstructure and associated attributes of metallic polycrystalline engineering materials in their bulk form. These methods are often combined with external stimuli such as thermo-mechanical loading to take snapshots of the evolving microstructure and attributes over time. However, the extreme data volumes and the high costs of traditional data acquisition and reduction approaches pose a barrier to quickly extracting actionable insights and improving the temporal resolution of these snapshots. This article presents a fully automated technique capable of rapidly detecting the onset of plasticity in high-energy X-ray microscopy data. The technique is computationally faster by at least 50 times than the traditional approaches and works for data sets that are up to nine times sparser than a full data set. This new technique leverages self-supervised image representation learning and clustering to transform massive data sets into compact, semantic-rich representations of visually salient characteristics ({\it e.g.} peak shapes). These characteristics can rapidly indicate anomalous events, such as changes in diffraction peak shapes. It is anticipated that this technique will provide just-in-time actionable information to drive smarter experiments that effectively deploy multi-modal X-ray diffraction methods spanning many decades of length scales.},
keywords = {high-energy diffraction microscopy, machine learning},
}


```
