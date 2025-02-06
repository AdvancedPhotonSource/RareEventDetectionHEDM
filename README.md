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
pip install jupyter
pip install opencv-python
pip install tqdm
pip install fabio
```

## Testing Data:

We have shared the raw (ge5) testing data at the path ```/home/beams/WZHENG/RareEventDetectionHEDM/example_dataset/raw```. If you have access to the APS machine, you should be able to access it.


## Usage

This code has been mostly tested with GE detector data at 1-ID. 

Step 0: process the raw HEDM images 
```shell
# run the image_processing jupyter notebooks
```

Step 1: train the BYOL encoder on a baseline dataset (e.g., zero load):
```shell
conda activate event_detection
cd code/BraggEmb_code/
python main.py -training_scan_file $baselinePATH -training_dark_file $baselinedarkPATH -zdim $i -bkgd $bkgd
# copy trained model if needed
cp $model_savedPATH$model_savedNAME $model_dstPATH$model_dstNAME${i}.pth
```

Step 2: calculate REI values for subsequent datasets (i.e., scans at different loads):
```shell 
cd EventDetection_code

# raw scan file-based mode (runs on raw ge5 files)
python detection4all.py\
      -file_mode 1\
      -baseline_scan $baselinePATH\
      -baseline_scan_dark $baselinedarkPATH\
      -testing_scan $testPATH\
      -testing_scan_dark $testdarkPATH\
      -trained_encoder $model_dstPATH$model_dstNAME${i}.pth\
      -thold $thold\
      -output_scv ${eva_resultPATH}d${i}/res-${thr}-${k}.csv

# patch file mode (only expert mode... runs on HDF files which only have patches)
python detection4all.py\
      -patch_mode 1
      -baseline_scan $baselinePATH\
      -trained_encoder $model_dstPATH$model_dstNAME${i}.pth\
      -testing_scan $testingPATH\
      -thold $thold\
      -output_csv ${eva_resultPATH}d${i}/res-${thr}-${k}.csv

```

Example:

We added some example datasets for the step 1 and step 2, for step 0, please contact the authors for the example input dataset (around 12-14 GB for each raw file) 

There is a example file processing notebook at code folder that can be tried.

Step 1 (the default #epochs is set to 100, please change it if needed, dark file input is optional)
```shell
conda activate event_detection
cd code/BraggEmb_code/
python main.py \
      -training_scan_file /home/beams/WZHENG/RareEventDetectionHEDM/example_dataset/raw/park_ss_ff_0MPa_000315.edf.ge5\
      -training_dark_file /home/beams/WZHENG/RareEventDetectionHEDM/example_dataset/raw/dark_before_000320.edf.ge5\
      -thold 100
```

Step 2 (please change the -emdmdl name based on #epochs in the previous step, dark file is optional)
```shell
cd EventDetection_code

# raw scan file-based mode (runs on raw ge5 files)
python detection4all.py -file_mode 1\
      -baseline_scan /home/beams/WZHENG/RareEventDetectionHEDM/example_dataset/raw/park_ss_ff_0MPa_000315.edf.ge5\
      -baseline_scan_dark /home/beams/WZHENG/RareEventDetectionHEDM/example_dataset/raw/dark_before_000320.edf.ge5\
      -testing_scan /home/beams/WZHENG/RareEventDetectionHEDM/example_dataset/raw/park_ss_ff_260MPa_000497.edf.ge5\
      -testing_scan_dark /home/beams/WZHENG/RareEventDetectionHEDM/example_dataset/raw/dark_before_000502.edf.ge5\
      -thold 100\
      -output_csv res-04-40.csv

# Or run them separately:
python baseline_pre.py  -file_mode 1\
      -baseline_scan /home/beams/WZHENG/RareEventDetectionHEDM/example_dataset/raw/park_ss_ff_0MPa_000315.edf.ge5\
      -baseline_scan_dark /home/beams/WZHENG/RareEventDetectionHEDM/example_dataset/raw/dark_before_000320.edf.ge5\
      -thold 100\

python testing_scan.py  -file_mode 1\
      -testing_scan /home/beams/WZHENG/RareEventDetectionHEDM/example_dataset/raw/park_ss_ff_260MPa_000497.edf.ge5\
      -testing_scan_dark /home/beams/WZHENG/RareEventDetectionHEDM/example_dataset/raw/dark_before_000502.edf.ge5\
      -thold 100\
      -output_csv res-04-40.csv

# patch file mode (only expert mode... runs on HDF files which only have patches)
python detection4all.py -patch_mode 1\
      -baseline_scan /home/beams/WZHENG/RareEventDetectionHEDM/example_dataset/patch/park_ss_ff_0MPa_000315.edf.h5\
      -testing_scan /home/beams/WZHENG/RareEventDetectionHEDM/example_dataset/patch/\
      -thold 100\
      -output_csv res-04-40.csv
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
