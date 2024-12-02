#! /bin/bash


# embedding model training related parameters and paths
embedsize=(16 32 64 128)

baselinePATH=/home/beams/WZHENG/EventDetection/park_datasets_from_hemant/
baselineNAME=park_ss_ff3_000315_to_318.edf.h5
train_wrkPath=/home/beams/WZHENG/BraggEmb/
model_savedPATH=/home/beams/WZHENG/BraggEmb/debug-itrOut/
model_savedNAME=script-ep00100.pth
model_dstPATH=/home/beams/WZHENG/EventDetection/modelTrained/
model_dstNAME=park_315_318-d

# event detection related parameters and paths
thresholds=$(seq 0.1 0.05 0.4)
numcenters=$(seq 10 50)

eva_datasetPATH=/home/beams/WZHENG/EventDetection/park_datasets_from_hemant/
eva_wrkPATH=/home/beams/WZHENG/EventDetection/rareEventHEDM_v1/
eva_resultPATH=/home/beams/WZHENG/EventDetection/rareEventHEDM_v1/full_parameter_test/


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