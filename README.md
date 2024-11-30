# RareEventDetectionHEDM
Rapid detection of rare events from in situ X-ray diffraction data using machine learning

<p float="left">
  <img src="doc/REI_schematic.png" width="250" />
  <img src="/doc/REI-detailed-schematic.png" width="250" /> 
</p>

## Installation

We recommend creating a new environment (through conda, pipenv, etc) to try out RareEventDetectionHEDM. The conda command might look like:
```shell
conda create -n event_detection python=3.9

conda activate event_detection

pip install torch
pip install numpy
pip install h5py
pip install torchvision
pip install scikit-learn
pip install pandas
```

## Usage

This code has been mostly tested with GE detector data at 1-ID. 

Step 0: process the raw HEDM images 
```shell
# run the image_processing jupyter notebooks
```

Step 1: train the BYOL encoder on a baseline dataset (e.g., zero load):
```shell
conda activate event_detection
cd BraggEmb/ 
python main.py -ih5 $baselinePATH$baselineNAME -zdim $i
cp $model_savedPATH$model_savedNAME $model_dstPATH$model_dstNAME${i}.pth
```

Step 2: calculate REI values for subsequent datasets (i.e., scans at different loads):
```shell
cd $eva_wrkPATH
python detection4all.py\
      -bh5 $baselinePATH$baselineNAME\
      -embmdl $model_dstPATH$model_dstNAME${i}.pth\
      -ids $eva_datasetPATH\
      -ocsv ${eva_resultPATH}d${i}/res-${thr}-${k}.csv\
      -uqthr=${thr} -ncluster=${k}
...
```

Example:
```shell
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
