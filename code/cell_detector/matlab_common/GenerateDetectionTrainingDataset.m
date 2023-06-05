function [images] = GenerateDetectionTrainingDataset(imagesPath,windowSize,predictionPatchSize,stride,valPercentage,cellType,features)
% GenerateDetectionTrainingDataset as name suggest
%   Inputs:
%       imagesPath: a cell of paths to folders containing original 
%                   rgb images and their ground truths.
%       windowSize : the size of an input window to CNN
%       predictionPatchSize : the size of an output path from CNN
%       stride : step size between consecutive input window on an original rgb image
%       valPercentage : the percentage of a validation data
%       cellType : type of cell to be detected
%       features : a cell for training features
%
%   Outputs:
%       images: a structure of a dataset for CNN
%           images.data - 4D matrix
%           images.data_mean 
%           images.labels - 4D matrix
%           images.set - 1 for traning, 2 for validation
%           images.showImage
%           images.lambda
%           images.colourMat (if features contain 'h')
%
% Korsuk Sirinukunwattana
% BIAlab, Department of Computer Science, University of Warwick
% 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
allImages = [];
allProximity = [];
group = zeros(length(imagesPath),1);

% calculate features and proximity
for iImage = 1:length(imagesPath)
    file = dir(fullfile(imagesPath{iImage},'*.bmp'));
    fprintf('Processing %s ... %2.2f %% complete \n', file.name, 100*iImage/length(imagesPath));
    I = imread(fullfile(imagesPath{iImage},file.name));  % read an RGB image
    I = Retinex(I);     % adjust an image  
    
    % feature
    feat = [];
    for iFeatures = 1:length(features)
        switch features{iFeatures}
            case 'rgb'
                feat = cat(3,feat,single(I));
            case 'lab'
                feat = cat(3,feat,single(rgb2lab(I)));
            case 'h'
                doubleRGB = RemoveArtefact(I);
          
                colourMat = EstUsingSCD(doubleRGB);
                [ DCh ] = Deconvolve( I, colourMat );
                [ H ] = PseudoColourStains( DCh, colourMat );
                H = rgb2gray(H);
                feat = cat(3,feat,single(H));
            case 'e'
                doubleRGB = RemoveArtefact(I);
                colourMat = EstUsingSCD(doubleRGB);
                [ DCh ] = Deconvolve( I, colourMat );
                [ ~, E ] = PseudoColourStains( DCh, colourMat );
                E = rgb2gray(E);
                feat = cat(3,feat,single(E));
            case 'he'
                doubleRGB = RemoveArtefact(I);
                colourMat = EstUsingSCD(doubleRGB);
                [ DCh ] = Deconvolve( I, colourMat );
                [ H, E ] = PseudoColourStains( DCh, colourMat );
                H = rgb2gray(H);
                E = rgb2gray(E);
                feat = cat(3,feat,single(H),single(E));
            case 'br'
                BR = BlueRatioImage(I);
                feat = cat(3,feat,single(BR));
            case 'grey'
                grey = rgb2gray(I);
                feat = cat(3,feat,single(grey));
        end
    end   
    allImages = cat(4,allImages,feat);
    
    % proximity
    if exist(fullfile(imagesPath{iImage},[file.name(1:end-4),'_',cellType,'.mat']),'file')
        J = load(fullfile(imagesPath{iImage},[file.name(1:end-4),'_',cellType,'.mat']));  % read a ground truth image
        J = J.detection;
        group(iImage) = 1;
    else
        J = [];
    end
        
    alpha = 0.5;
    M = GenerateDetectionGTProximity(J,5,alpha,[size(feat,1),size(feat,2)]);           % r = 4, 
    allProximity = cat(4,allProximity,M);
end

% generate training and test data
cv = cvpartition(group,'HoldOut',valPercentage/100);
trainIdx = training(cv,1);
valIdx = test(cv,1);

a = find(group&valIdx);
b = find(valIdx);
if isempty(setdiff(b,a))
    valIdx = a;
else
    valIdx = cat(1,a,setdiff(b,a));
end

trainFeats = allImages(:,:,:,trainIdx);
valFeats =  allImages(:,:,:,valIdx);
trainHandmark = allProximity(:,:,:,trainIdx);
valHandmark = allProximity(:,:,:,valIdx);

%save('Break.mat');
% images
[images] = PrepareDetectionTrainValDataset(windowSize,predictionPatchSize,stride,trainFeats,valFeats,trainHandmark,valHandmark);
images.features = features;

end
