#!/bin/bash

currentPath=$(dirname "$0")
CSVAnnotationsFolder="/../../../training_data/input_data/cell_labels/"
TilePath="../../../training_data/input_data/annotation_tiles/"
WarMatPath="../../../training_data/detection_data/WarMat/"
OutMatPath="../../../training_data/detection_data/OutMat/"
h5Path="../../../training_data/detection_data/h5/"
NegativeLabels="['unk']"
trainH5Name="TrainData.h5"
validH5Name="ValidData.h5"

PositiveWarMatPath="${WarMatPath}/Positive/"
NegativeWarMatPath="${WarMatPath}/Negative/"

matlabPath="${currentPath}/../matlab_common/"

mkdir -p "$PositiveWarMatPath" "$NegativeWarMatPath" "$OutMatPath" "$h5Path"

matlab -nodesktop -nosplash -r "addpath(genpath('$currentPath'), genpath('$matlabPath')); S0('$CSVAnnotationsFolder', '$TilePath', '$PositiveWarMatPath', '$NegativeWarMatPath', $NegativeLabels); S1('$PositiveWarMatPath', '$OutMatPath', 'p'); S1('$NegativeWarMatPath', '$OutMatPath', 'n'); exit;"

python3 "${currentPath}/S2.py" "$OutMatPath" "$h5Path" "$trainH5Name" "$validH5Name"
