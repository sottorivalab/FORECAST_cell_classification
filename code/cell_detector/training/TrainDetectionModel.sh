#!/bin/bash

currentPath=$(dirname "$0")
TrainingCodePath="${currentPath}/../analysis"
ModelPath="${currentPath}/../../models/detection/"
TrainingDataPath="${currentPath}/../../training_data/detection_patches/"
TrainDataFilename="TrainData.h5"
ValidDataFilename="ValidData.h5"

python3 "${TrainingCodePath}/Run_training.py" "$ModelPath" "$TrainingDataPath" "$TrainDataFilename" "$ValidDataFilename"
