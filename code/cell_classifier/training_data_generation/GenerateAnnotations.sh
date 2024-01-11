#!/bin/bash

currentPath=$(dirname "$0")
CSVAnnotationsFolder="/../../../training_data/input_data/cell_labels/"
TilePath="../../../training_data/input_data/annotation_tiles/"
OutPath="../../../training_data/classification_patches/"
nPatches=10000
patchSize="[51, 51]"
cellLabels="{'epithelial', 'stromal', 'immune', 'unk'}"
nAngles=4
maxShift=5

PositiveWarMatPath="${WarMatPath}/Positive/"
NegativeWarMatPath="${WarMatPath}/Negative/"

mkdir -p "$OutPath"

matlab -nodesktop -nosplash -r "addpath(genpath('$currentPath')); GenerateImagePatchesFromCSVs('$CSVAnnotationsFolder', '$TilePath', '$OutPath', $nPatches, $PatchSize, $cellLabels, $nAngles, $maxShift); exit;"
