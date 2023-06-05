function [images] = PrepareDetectionTrainValDataset(windowSize,predictionPatchSize,stride,trainFeats,valFeats,trainHandmark,valHandmark)
% PrepareDetectionTrainValDataset prepares training and validation data by extract
% small patches from a training image and a validation image and pack them
% in a struct.
%
%   Inputs:
%       windowSize: [nrow,ncol] of a patch
%       predictionPatchSize : [nrow,ncol] of a predicted patch
%                             Note that sizePredictionPatch should be smaller
%                             than or equal to the size of sizeWindow
%       stride: [step row, step col] between consecutive patch.
%       trainFeats: a 4-D rgb image for training. The 4th D corresponds
%                   to an individual image
%       trainHandmark: a 4-D handmark image for training. The 4rd D corresponds
%                      to an individual image
%       valFeats: a 4-D rgb image for validation
%       valHandmark: a 4-D handmark image for validation
%
%   Outputs:
%       images: a structure with the following fields.
%           'data' a 4D matrix. 1st and 2nd D are spatial dimension. 3rd D
%                  is image features, and 4th D corresponds to an individual
%                  image.
%           'data_mean' an vector of means across the 3rd and 4th
%                       dimensions
%           'labels' a 4D matrix for ground truth images. The 4rd D
%                    corresponds to an individual image
%           'set' a numerical flag indicating a type of data.
%                 1 = train, 2 = validation
%           'showImage': an image used to monitor the convergence of the
%                         network
%           'lambda': a loss penalty parameter for regression
% 
% Korsuk Sirinukunwattana
% BIAlab, Department of Computer Science, University of Warwick
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
images = struct('data',[],'data_mean',[],'labels',[],'set',[],'showImage',[],'lambda',[]);

% extract patches from training and validation images
dataTrain = [];
dataVal = [];
labelsTrain = [];
labelsVal = [];

for i = 1:size(trainFeats,4)
    [dTrain,lTrain] = ExtractPatch_train(windowSize,predictionPatchSize,stride,trainFeats(:,:,:,i),trainHandmark(:,:,:,i));
    dataTrain = cat(4,dataTrain,dTrain);
    labelsTrain = cat(4,labelsTrain,lTrain);
end

for i = 1:size(valFeats,4)
    [dVal,lVal] = ExtractPatch_train(windowSize,predictionPatchSize,stride,valFeats(:,:,:,i),valHandmark(:,:,:,i));
    dataVal = cat(4,dataVal,dVal);
    labelsVal = cat(4,labelsVal,lVal);
end

images.data = cat(4,dataTrain,dataVal);
images.labels = cat(4,labelsTrain,labelsVal);
images.data_mean = mean(mean(mean(dataTrain,4),1),2);   % calculate mean from training data

% Matconvnet needs an input of single class
images.data = single(images.data);          
images.labels = single(images.labels);     
images.data_mean = single(images.data_mean);

% preprocess data
% images.data = bsxfun(@minus,images.data,images.data_mean);

% train = 1, validation = 2
set = cat(1,ones(size(dataTrain,4),1),2*ones(size(dataVal,4),1));
images.set = set;

% displayed image
images.showImage = valFeats(:,:,:,1);
if size(images.showImage,1) > 200
    nrow = 200;
else
    nrow = size(images.showImage,1);
end
if size(images.showImage,2) > 400
    ncol = 400;
else
    ncol = size(images.showImage,2);
end

images.showImage = images.showImage(1:nrow,1:ncol,:);

% lambda
flag = labelsTrain(:)>0;
images.lambda = sum(flag)/sum(~flag);

end
