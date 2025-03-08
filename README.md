# RareEventDetectionHEDM
Rapid detection of rare events from in situ X-ray diffraction data using machine learning. In particular, we applied this method to high-energy diffraction microscopy (HEDM). In a typical in situ HEDM experiment, the macroscopic stimulus to be applied on a sample is decided a priori, on the basis of a known response relationship. For instance, in an in situ HEDM experiment where a sample is subject to various levels of mechanical loading to study the material response heterogeneity at the mesoscale, the levels at which the loading is paused – to deploy a higher-resolution and more beamtime-consuming characterization method – are often decided according to macroscopic milestones such as yield strength in the macroscopic stress–strain curve of the material. Here, we present a robust, self-supervised, machine learning based framework that enables rapid identification of, and thus automated response to, the minute changes in diffraction spots measured by FF-HEDM and probable microstructural changes in polycrystalline metals. We call it "rare event" because while yielding in a uniaxial tension experiment is an eventuality but a "rare event" with regard to when and where it occurs. 

Here is this all works. In a typical FF-HEDM experiment, the sample is rotated and diffraction images are taken at each rotation angle. At APS 1-ID, 1440 fames is typicaly for one HEDM scan (i.e., a full rotation) of the sample. Scans are then repeated as the load on the sample is increased. Currently, each scan is at one fixed load point. (One can think about doing variable load in one scan, but that's for the future work!)  In this work, we are using a 4-megapixel GE a-Si flat panel detector. (We have NOT test this on a modular detector such as the Dectris Pilatus which have gaps! This should work on other flat panel detector which with no gaps such as the Varex flat panels.) At APS, we use the ge5 file format, but this code use the fabIO package which should be able to read a wide variety of X-ray detector data formats. 

Let's talk about the details of the workflow now: 
* **Training:** The user needs to decide which scan is to consider as the "baseline" scan. (Note: in the paper, we talk about the baseline and reference datasets. Typically, these are the same thing (i.e., the zero load scan), but in principle can be different.)  This is typically when the sample is under no load. So, a user needs to take the baseline dataset and use it to train the model. The first steps to take full detector images (i.e., dark subtracted and lower thresholded) and find all the Bragg peaks and create a so-called patches (currently patch size is 15x15 pixels hardcoded). We use connected components method for this. Once we have the pacthes, we need to do the training which has 2 steps: training the encoder and find the K-centers. This encoder is basically a latent space representation of the Bragg peaks. We use the BYOL method. This method learns the latent features of the Bragg peaks in the baseline dataset and trains and encoder. This  encoder generates a 1-D representation vector of the Bragg peaks (i.e., sometimes called the "embedding"). Next, we user a K-means clustering algorithm to group these representation vectors into K centers. This clustering process allows us to identify common patterns and clusters within the baselin dataset. The user needs to pick the number of clusters; we have that 30 is good number to use.  
* **al:**
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
