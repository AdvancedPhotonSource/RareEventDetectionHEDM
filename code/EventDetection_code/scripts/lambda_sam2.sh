#!/bin/bash

eval "$(conda shell.bash hook)"
conda env list
conda activate fairDMS

# embedding model training related parameters and paths
embedsize=(64)

baselinePATH=/homes/wzheng/kasemer_mar22_patches/kasemer_sam2_patch_dir/
baselineNAME=sam2_ff_0N_000027.edf.h5
train_wrkPath=/homes/wzheng/BraggEmb/
exp_name=sam2
model_savedPATH=/homes/wzheng/BraggEmb/${exp_name}-itrOut/
model_savedNAME=script-ep00100.pth
model_dstPATH=/homes/wzheng/EventDetection/modelTrained/
model_dstNAME=sam2_ff_000027-d

# event detection related parameters and paths
thresholds=$(seq 0.3 0.05 0.35)
numcenters=$(seq 25 5 30)

eva_datasetPATH=/homes/wzheng/kasemer_mar22_patches/kasemer_sam2_patch_dir/
eva_wrkPATH=/homes/wzheng/EventDetection/rareEventHEDM_v1/
eva_resultPATH=/homes/wzheng/EventDetection/rareEventHEDM_v1/sam2_test/

# module load conda
# conda activate fairDMS

# first we need to train the model using different embedding dimensions

for i in "${embedsize[@]}"
do
  cd $train_wrkPath
  python main.py -expName $exp_name -ih5 $baselinePATH$baselineNAME -zdim $i -gpus 6
  # rename the model to 'park_315_318.pth' and save it to the path '../EventmodelTraned'
  cp $model_savedPATH$model_savedNAME $model_dstPATH$model_dstNAME${i}.pth
done

echo 'finished generating models, now need to test'

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
      -embmdl $model_dstPATH$model_dstNAME${i}.pth\
      -ids $eva_datasetPATH\
      -ocsv ${eva_resultPATH}d${i}/res-${thr}-${k}.csv\
      -uqthr=${thr} -ncluster=${k}\
      -dpre $exp_name
     done
  done 
done
