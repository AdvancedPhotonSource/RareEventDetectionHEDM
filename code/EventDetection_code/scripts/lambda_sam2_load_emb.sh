#!/bin/bash

eval "$(conda shell.bash hook)"
conda env list
conda activate fairDMS

# this script is for using the embedding model trained on park 22 to test on sam2

# embedding model training related parameters and paths
embedsize=(64)

baselinePATH=/homes/wzheng/kasemer_mar22_patches/kasemer_sam2_patch_dir/
baselineNAME=sam2_ff_0N_000027.edf.h5
train_wrkPath=/homes/wzheng/BraggEmb/
exp_name=sam2

model_dstPATH=/homes/wzheng/EventDetection/modelTrained/
model_dstNAME=park_00375.pth

# event detection related parameters and paths
thresholds=$(seq 0.3 0.05 0.35)
numcenters=$(seq 25 5 30)

eva_datasetPATH=/homes/wzheng/kasemer_mar22_patches/kasemer_sam2_patch_dir/
eva_wrkPATH=/homes/wzheng/EventDetection/rareEventHEDM_v1/
eva_resultPATH=/homes/wzheng/EventDetection/rareEventHEDM_v1/sam2_test_pretrained/

# module load conda
# conda activate fairDMS

echo 'now need to test use the pretrained model'

mkdir $eva_resultPATH
cd $eva_wrkPATH

for i in "${embedsize[@]}"
do
  mkdir ${eva_resultPATH}d${i}/
  for thr in $thresholds
  do
    for k in ${numcenters[@]}
    do
      python detection4all.py\
      -bh5 $baselinePATH$baselineNAME\
      -embmdl $model_dstPATH$model_dstNAME\
      -ids $eva_datasetPATH\
      -ocsv ${eva_resultPATH}d${i}/res-${thr}-${k}.csv\
      -uqthr=${thr} -ncluster=${k}\
      -dpre $exp_name
     done
  done 
done
