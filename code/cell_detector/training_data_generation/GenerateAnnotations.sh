#!/bin/bash

currentPath=$(dirname "$0")
CSVAnnotationsFolder=$1
TilePath=$2
WarMatPath=$3
OutMatPath=$4
h5Path=$5
NegativeLabels=$6
trainH5Name=$7
validH5Name=$8

PositiveWarMatPath="${WarMatPath}/Positive/"
NegativeWarMatPath="${WarMatPath}/Negative/"

matlabPath="${currentPath}/../matlab_common/"

mkdir -p "$PositiveWarMatPath" "$NegativeWarMatPath" "$OutMatPath" "$h5Path"

matlab -nodesktop -nosplash -r "addpath(genpath('$currentPath'), genpath('$matlabPath')); S0('$CSVAnnotationsFolder', '$TilePath', '$PositiveWarMatPath', '$NegativeWarMatPath', $NegativeLabels); S1('$PositiveWarMatPath', '$OutMatPath', 'p'); S1('$NegativeWarMatPath', '$OutMatPath', 'n'); exit;"

python3 "${currentPath}/S2.py" "$OutMatPath" "$h5Path" "$trainH5Name" "$validH5Name"
