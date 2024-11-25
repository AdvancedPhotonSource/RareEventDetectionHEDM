#!/bin/bash

eval "$(conda shell.bash hook)"
conda env list
conda activate vit_bnn_test

# this script is for using the embedding model trained on 


# embedding model training related parameters and paths
embedsize=(64)


baselinePATH=/home/wzheng/EventDetection/sangid_patch_dir_case2/
baselineNAME=lshr_r1e_state_2_bigbox_ff_000504.h5
train_wrkPath=/home/wzheng/WZHENG/BraggEmb/
model_savedPATH=/home/wzheng/BraggEmb/debug-itrOut/
model_savedNAME=script-ep00100.pth
model_dstPATH=/home/wzheng/EventDetection/modelTrained/
model_dstNAME=lshr_504_emb
exp_name=lshr

model_dstPATH=/home/wzheng/EventDetection/modelTrained/
model_dstNAME=lshr_504_emb64.pth

# event detection related parameters and paths
thresholds=$(seq 0.2 0.1 0.7)
numcenters=$(seq 30 10 60)

eva_datasetPATH=/home/wzheng/EventDetection/sangid_patch_dir_case2/
eva_wrkPATH=/home/wzheng/EventDetection/rareEventHEDM_v1/
eva_resultPATH=/home/wzheng/EventDetection/rareEventHEDM_v1/case2/


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
  mkdir ${eva_resultPATH}d${i}${exp_name}/
  for thr in $thresholds
  do
    for k in ${numcenters[@]}
    do
      python detection4all.py\
      -bh5 $baselinePATH$baselineNAME\
      -embmdl $model_dstPATH$model_dstNAME\
      -ids $eva_datasetPATH\
      -ocsv ${eva_resultPATH}d${i}${exp_name}/res-${thr}-${k}.csv\
      -uqthr=${thr} -ncluster=${k}\
      -dpre $exp_name\
      -degs 360 > lshr_0606_case2.log
    done
  done 
done