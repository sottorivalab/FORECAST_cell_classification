#!/bin/bash

tilePath="../tiles/tiles_norm_c/"
segmentationTilePath="../tiles/masks/"
cellDetectionResultsPath="../results/detection/"
cellClassificationResultsPath="../results/classificaion/"

cellDetectorCheckPointPath="../models/detection/"
cellClassifierPath="../models/classification/Cell_Classifier.h5"

labelFile="../config/cell_labels.txt"
labelNames="['epithelial', 'stromal', 'immune', 'immune', 'unknown']"
noLabelIdx=4

detectionBatchSize=500
classificationBatchSize=50
cellClassCertainty=0.0
outputProbs=False

if [ $# -gt 0 ]; then
    all_files=("$tilePath"/*/)
    files=()
    for var in "$@"; do
        files+=("${all_files[$((var-1))]}")
    done
else
    files=("$tilePath"/*/)
fi

currentPath=$(dirname "${0}")

sccnnDetectionCodePath="${currentPath}/cell_detector/analysis/"
classificationCodePath="${currentPath}/cell_classifier/classification/"
matlabPath="${currentPath}/cell_detector/matlab_common/"
outputAnnotationCodePath="${currentPath}/cell_classifier/output_image_labelling/"
mergeCSVCodePath="${currentPath}/cell_classifier/merge_csvs/"

cellDetectionCSVPath="${cellDetectionResultsPath}/20180117/csv/"
cellClassificationCSVPath="${cellClassificationResultsPath}/csv/"
smallDotTilePath="${cellClassificationResultsPath}/labelledImages/"
bigDotTilePath="${cellClassificationResultsPath}/labelledImagesBigDot/"
tifPath="${cellClassificationResultsPath}/tif/"
mergeCSVPath="${cellClassificationResultsPath}/all_cells/"

for file in "${files[@]}"; do
    imageName="$(basename "$file")"
    
    imageWidth=$(sed -n 's/iWidth=//p' "${tilePath}/${imageName}/FinalScan.ini" | head -1)
    imageHeight=$(sed -n 's/iHeight=//p' "${tilePath}/${imageName}/FinalScan.ini" | head -1)
    tileWidth=$(sed -n 's/iImageWidth=//p' "${tilePath}/${imageName}/FinalScan.ini")
    tileHeight=$(sed -n 's/iImageHeight=//p' "${tilePath}/${imageName}/FinalScan.ini")

    (cd "${sccnnDetectionCodePath}" && source activate tf1p4 && python3 "./Generate_Output.py" "${cellDetectorCheckPointPath}" "${tilePath}" "${cellDetectionResultsPath}" "${detectionBatchSize}" "${imageName}" "${segmentationTilePath}")

    source activate pytorch0p3

    python3 -c "import sys; sys.path.append('${classificationCodePath}'); import processCSVs; processCSVs.processCSVs('${imageName}', '${cellDetectionCSVPath}', '${tilePath}', '${cellClassifierPath}', '${cellClassificationCSVPath}', segmentPath='${segmentationTilePath}', batchSize=${classificationBatchSize}, outLabels=${labelNames}, minProb=${cellClassCertainty}, noClassLabel=${noLabelIdx}, outputProbs=${outputProbs})"

    matlabOpeningCommands="addpath(genpath('${matlabPath}'), genpath('${outputAnnotationCodePath}'), genpath('${mergeCSVCodePath}'));"

    dotAnnotationSize=6
    tifFile="${tifPath}/${imageName%.*}_Annotated.tif"

    matlabSmallDotCommands="WriteAnnotations('${imageName}', '${cellClassificationCSVPath}', '${tilePath}', '${smallDotTilePath}', '${labelFile}', ${dotAnnotationSize}); Tiles2TIF('${smallDotTilePath}/${imageName}/', [${tileWidth} ${tileHeight}], [${imageWidth}, ${imageHeight}], '${tifFile}', 'jpg', false);"

    dotAnnotationSize=30
    tifFile="${tifPath}/${imageName%.*}_AnnotatedBigDot.tif"

    matlabBigDotCommands="WriteAnnotations('${imageName}', '${cellClassificationCSVPath}', '${tilePath}', '${bigDotTilePath}', '${labelFile}', ${dotAnnotationSize}); Tiles2TIF('${bigDotTilePath}/${imageName}/', [${tileWidth} ${tileHeight}], [${imageWidth}, ${imageHeight}], '${tifFile}', 'jpg', false);"
    
    mergeCSVFile="${mergeCSVPath}/${imageName%.*}.csv"
    
    matlabMergeCSVCommands="MergeCSVs('${cellClassificationCSVPath}/${imageName}', '${tilePath}/${imageName}', '${mergeCSVFile}');"

    matlab -nodesktop -nosplash -r "${matlabOpeningCommands} ${matlabSmallDotCommands} ${matlabBigDotCommands} ${matlabMergeCSVCommands} exit;"
done
