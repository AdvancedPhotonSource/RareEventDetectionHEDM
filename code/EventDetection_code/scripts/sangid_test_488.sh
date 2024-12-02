#!/bin/bash

eval "$(conda shell.bash hook)"
conda env list
conda activate event_detection

# this script is for using the embedding model trained on 


# embedding model training related parameters and paths
embedsize=(64)


baselinePATH=/home/beams/WZHENG/EventDetection/sangid_patch_dir/
baselineNAME=lshr_r1e_bigbox_ff_000488.h5
train_wrkPath=/home/beams/WZHENG/BraggEmb/
model_savedPATH=/home/beams/WZHENG/BraggEmb/debug-itrOut/
model_savedNAME=script-ep00100.pth
model_dstPATH=/home/beams/WZHENG/EventDetection/modelTrained/
model_dstNAME=sangid_00482-d
exp_name=lshr

model_dstPATH=/home/beams/WZHENG/EventDetection/modelTrained/
model_dstNAME=sangid_00488.pth

# event detection related parameters and paths
thresholds=$(seq 0.15 0.05 0.5)
numcenters=$(seq 10 5 50)

eva_datasetPATH=/home/beams/WZHENG/EventDetection/sangid_patch_dir/
eva_wrkPATH=/home/beams/WZHENG/EventDetection/rareEventHEDM_v1/
eva_resultPATH=/home/beams/WZHENG/EventDetection/rareEventHEDM_v1/


# first we need to train the model using different embedding dimensions
# for i in "${embedsize[@]}"
# do
#   cd $train_wrkPath
#   python main.py -ih5 $baselinePATH$baselineNAME -zdim $i
#   # rename the model to 'park_315_318.pth' and save it to the path '../EventmodelTraned'
#   cp $model_savedPATH$model_savedNAME $model_dstPATH$model_dstNAME${i}.pth
# done

echo 'finished generating models, now need to test'
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
      -degs 360 > sangid_00488.log
    done
  done 
done