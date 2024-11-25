#!/bin/bash -l

#PBS -A hp-ptycho
#PBS -l walltime=02:00:00
#PBS -l filesystems=home
#PBS -N polaris_event_test
#PBS -q preemptable

# embedding model training related parameters and paths
embedsize=(64)

baselinePATH=/home/wzheng/testing_dataset/
baselineNAME=park_ss_ff3_000315_to_318.edf.h5
train_wrkPath=/home/wzheng/BraggEmb/
model_savedPATH=/home/wzheng/BraggEmb/debug-itrOut/
model_savedNAME=script-ep00100.pth
model_dstPATH=/home/wzheng/EventDetection/modelTrained/
model_dstNAME=park_315_318-d

# event detection related parameters and paths
thresholds=$(seq 0.3 0.05 0.35)
numcenters=$(seq 25 5 30)

eva_datasetPATH=/home/wzheng/testing_dataset/
eva_wrkPATH=/home/wzheng/EventDetection/rareEventHEDM_v1/
eva_resultPATH=/home/wzheng/EventDetection/rareEventHEDM_v1/polaris_test/

module load conda
conda activate event_detection

# first we need to train the model using different embedding dimensions

for i in "${embedsize[@]}"
do
  cd $train_wrkPath
  python main.py -ih5 $baselinePATH$baselineNAME -zdim $i
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
      -uqthr=${thr} -ncluster=${k}
     done
  done 
done
