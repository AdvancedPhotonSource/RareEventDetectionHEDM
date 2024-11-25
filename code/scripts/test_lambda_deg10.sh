#!/bin/bash

eval "$(conda shell.bash hook)"
conda env list
conda activate fairDMS

# this script is for using the embedding model trained on park 22 to test on sam3

# embedding model training related parameters and paths
embedsize=(64)

baselinePATH=/homes/wzheng/EventDetection/test_patches_v1/
baselineNAME=park_ss_ff3_000375.edf.h5
train_wrkPath=/homes/wzheng/BraggEmb/
exp_name=park_ss_ff

model_dstPATH=/homes/wzheng/EventDetection/modelTrained/
model_dstNAME=park_00375.pth

# event detection related parameters and paths
thresholds=$(seq 0.3 0.05 0.3)
numcenters=$(seq 25 5 25)

eva_datasetPATH=/homes/wzheng/EventDetection/test_patches_v1/
eva_wrkPATH=/homes/wzheng/EventDetection/rareEventHEDM_v1/
eva_resultPATH=/homes/wzheng/EventDetection/rareEventHEDM_v1/degree10_test_seed1/

# module load conda
# conda activate fairDMS

# first we need to train the model using different embedding dimensions

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
      -dpre $exp_name\
      -degs 10\
      -seed 1 > deg10_seed1.log &
    done
  done 
done
