import sys
import math
import os
import glob as GLOB
import csv
import cv2
import numpy as np
import torch
import scipy.io as sio
import time
from fastai.transforms import *
from fastai.conv_learner import *

def processCSVs(imagePath, detectionPath, tilePath, classifierPath, outPath, segmentPath=None, windowSize=[51, 51], cellImageSize=224, inLabels=None, outLabels=['nep', 'unk', 'myo', 'cep', 'fib', 'lym', 'neu', 'mac', 'end'], batchSize=30, arch=dn201, gpu=True, overwrite=False, minProb=0.0, noClassLabel=None, outputProbs=False):
    if arch is not None:
        tforms = tfms_from_model(arch, cellImageSize)
        
    model = torch.load(classifierPath, map_location=lambda storage, loc: storage)
    
    if gpu:
        nTries = 10

        for i in range(nTries):
            try:
                model = model.cuda().eval()
            except:
                if i==nTries-1:
                    print('Cuda failed to load on attempt '+str(i+1)+', giving up.')
                    raise
                else:
                    print('Cuda failed to load on attempt '+str(i+1)+', retrying')
                    time.sleep(10)
    else:
        model = model.cpu().eval()
    
    detectionPath = os.path.join(detectionPath, '')

    CSVs = GLOB.glob(os.path.join(detectionPath, imagePath, '*.csv'))

    for csvPath in CSVs:
        outCSVPath = os.path.join(outPath, csvPath[len(detectionPath):])
        
        if outputProbs:
            outProbCSVPath = os.path.join(os.path.dirname(outCSVPath), 'probabilities', os.path.basename(outCSVPath))
        
        if overwrite or not os.path.isfile(outCSVPath):
            print("Classifying cells in: "+csvPath, flush=True)
            tileImagePath = os.path.join(tilePath, csvPath[len(detectionPath):-4]+'.jpg')
            tileImage = open_image(tileImagePath)

            tileImage = np.concatenate((tileImage[int(math.floor((windowSize[1]-1)/2))::-1,:], tileImage, tileImage[:int(math.floor(-(windowSize[1]-1)/2)):-1,:]),0)
            tileImage = np.concatenate((tileImage[:,int(math.floor((windowSize[1]-1)/2))::-1], tileImage, tileImage[:,:int(math.floor(-(windowSize[1]-1)/2)):-1]),1)
            
            if segmentPath is not None and len(segmentPath) > 0:
                segmentImagePath = os.path.join(segmentPath, csvPath[len(detectionPath):-4]+'.png')
                
                if os.path.isfile(segmentImagePath):
                    segmentImage = cv2.imread(segmentImagePath, cv2.IMREAD_GRAYSCALE)
                else:
                    segmentImage = None
            else:
                segmentImage = None
            
            with open(csvPath, newline='') as csvFile:
                csvReader = csv.reader(csvFile, delimiter=',')
                csvData = list(csvReader)
            
                os.makedirs(os.path.dirname(outCSVPath), exist_ok=True)
                
                if outputProbs:
                    os.makedirs(os.path.dirname(outProbCSVPath), exist_ok=True)
                
                csvHeader = csvData[0]
                
                if segmentImage is not None:
                    csvData = [csvData[i] for i in range(1, len(csvData)) if segmentImage[int(csvData[i][2])-1, int(csvData[i][1])-1] > 0]
                else:
                    csvData = csvData[1:]
                
                if outputProbs:
                    cellProbs = np.zeros((len(csvData), len(outLabels)+2))
                    for i in range(0, len(csvData)):
                        cellProbs[i, -2:] = csvData[i][1:]
                
                if inLabels is not None:
                    validRows = [i for i in range(0, len(csvData)) if csvData[i][0] in inLabels]
                else:
                    validRows = range(0, len(csvData))
                
                cellLabels = np.zeros(len(validRows))
                
                for i in range(0, len(validRows), batchSize):
                    iEnd = min(i+batchSize, len(validRows))
                    
                    cellImages = np.zeros((iEnd-i, 3, cellImageSize, cellImageSize))
                        
                    for j in range(0, iEnd-i):
                        cellImageRangeY = [int(csvData[validRows[i+j]][1]), int(csvData[validRows[i+j]][1])+windowSize[1]]
                        cellImageRangeX = [int(csvData[validRows[i+j]][2]), int(csvData[validRows[i+j]][2])+windowSize[0]]
                        cellImage = tileImage[cellImageRangeX[0]:cellImageRangeX[1], cellImageRangeY[0]:cellImageRangeY[1], :]
                        if arch is not None:
                            cellImages[j, :, :, :] = tforms[0](cellImage)
                        else:
                            cellImages[j, :, :, :] = cv2.resize(cellImage, (cellImageSize, cellImageSize), interpolation = cv2.INTER_AREA).transpose([2, 0, 1])
                
                    if gpu:
                        logProbs = model(Variable(torch.from_numpy(cellImages).cuda()).float()).cpu().data.numpy()
                    else:
                        logProbs = model(Variable(torch.from_numpy(cellImages)).float()).data.numpy()
                
                    probs = np.exp(logProbs)
                    batchLabels = np.argmax(probs, axis=1)
                    
                    if minProb > 0.0 and noClassLabel >= 0 and noClassLabel < len(outLabels):
                        batchLabels[np.max(probs, axis=1) < minProb] = noClassLabel;

                    cellLabels[i:iEnd] = batchLabels
                    
                    if outputProbs:
                        cellProbs[validRows[i:iEnd], :-2] = probs
                    
                for i in range(0, len(validRows)):
                    csvData[validRows[i]][0] = outLabels[int(cellLabels[i])]
            
                csvData = [csvHeader]+csvData
                
                with open(outCSVPath, 'w', newline='') as outCSVFile:
                    csvWriter = csv.writer(outCSVFile, delimiter=',')
                    csvWriter.writerows(csvData)
                    
                if outputProbs:
                    cellProbs = [cellProbs[i, :] for i in range(0, cellProbs.shape[0])]
                    cellProbs = [outLabels + ['x', 'y']] + cellProbs
                    
                    with open(outProbCSVPath, 'w', newline='') as outProbCSVFile:
                        csvProbWriter = csv.writer(outProbCSVFile, delimiter=',')
                        csvProbWriter.writerows(cellProbs)
        else:
            print("CSV file at "+outCSVPath+" already exists, skipping.", flush=True)
