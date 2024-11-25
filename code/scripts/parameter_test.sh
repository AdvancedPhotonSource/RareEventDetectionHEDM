#!/bin/bash

eval "$(conda shell.bash hook)"
conda env list
conda activate event_detection

# this script is for using the embedding model trained on park 22


# embedding model training related parameters and paths
embedsize=(64)


baselinePATH=/home/beams0/WZHENG/EventDetection/parameter_test_patches_315_372/
baselineNAME=park_ss_ff_000315_372.edf.h5
train_wrkPath=/home/beams0/WZHENG/BraggEmb/
model_savedPATH=/home/beams0/WZHENG/BraggEmb/debug-itrOut/
model_savedNAME=script-ep00100.pth
model_dstPATH=/home/beams0/WZHENG/EventDetection/modelTrained/
model_dstNAME=park_315_372-d
exp_name=park_ss_ff

model_dstPATH=/home/beams0/WZHENG/EventDetection/modelTrained/
model_dstNAME=park_00315_372_d64.pth

# event detection related parameters and paths
thresholds=$(seq 0.1 0.05 0.4)
numcenters=$(seq 10 5 40)

eva_datasetPATH=/home/beams0/WZHENG/EventDetection/parameter_test_patches_315_372/
eva_wrkPATH=/home/beams0/WZHENG/EventDetection/rareEventHEDM_v1/
eva_resultPATH=/home/beams0/WZHENG/EventDetection/rareEventHEDM_v1/beamsize_test_315_372_test_411_426/

# module load conda
# conda activate fairDMS

# first we need to train the model using different embedding dimensions

# first we need to train the model using different embedding dimensions

for i in "${embedsize[@]}"
do
  cd $train_wrkPath
  python main.py -ih5 $baselinePATH$baselineNAME -zdim $i
  # rename the model to 'park_315_318.pth' and save it to the path '../EventmodelTraned'
  cp $model_savedPATH$model_savedNAME $model_dstPATH$model_dstNAME${i}.pth
done

echo 'finished generating models, now need to test'
echo 'now need to test use the pretrained model'

# mkdir $eva_resultPATH
# cd $eva_wrkPATH

# for i in "${embedsize[@]}"
# do
#   mkdir ${eva_resultPATH}d${i}/
#   for thr in $thresholds
#   do
#     for k in ${numcenters[@]}
#     do
#       python detection4all.py\
#       -bh5 $baselinePATH$baselineNAME\
#       -embmdl $model_dstPATH$model_dstNAME\
#       -ids $eva_datasetPATH\
#       -ocsv ${eva_resultPATH}d${i}/res-${thr}-${k}.csv\
#       -uqthr=${thr} -ncluster=${k}\
#       -dpre $exp_name\
#       -degs 360 > deg360_315_372_beamsize_411_426.log &
#     done
#   done 
# done
