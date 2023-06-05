function [data,labels] = ExtractPatch(sizeWindow,sizePredictionPatch,stride,img,proximity)
% ExtractPatch extracts patch from an input image and its corresponding
% ground truth image
% Inputs:
%   sizeWindow: [nrow,ncol] of a patch
%   sizePredictionPatch : [nrow,ncol] of a predicted patch
%                         Note that sizePredictionPatch should be smaller
%                         than or equal to the size of sizeWindow
%   stride: [step row, step col]
%   img: an input image
%   proximity: ground truth proximity map
% Outputs:
%   data: a 4D matrix. The first two dimensions are the spatial dimension.
%         The 3rd D is the image feature. The 4th D contain each individual patch.
%   labels: a 4D matrix. The first two dimensions are the spatial
%           dimension.  The 3rd D is the image feature. The 4th D contain 
%           each individual patch.

% Korsuk Sirinukunwattana
% BIAlab, Department of Computer Science
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sizeImage = [size(img,1),size(img,2)];

idxWin = SlidingWindowIndex(sizeImage,sizeWindow,stride);

centerIndexX(1) = (sizeWindow(1) - sizePredictionPatch(1))/2;
centerIndexX(2) = sizeWindow(1) - centerIndexX(1);
centerIndexX(1) = centerIndexX(1) + 1;
centerIndexY(1) = (sizeWindow(2) - sizePredictionPatch(2))/2;
centerIndexY(2) = sizeWindow(2) - centerIndexY(1);
centerIndexY(1) = centerIndexY(1) + 1;

idxProx = reshape(idxWin,sizeWindow(1),sizeWindow(2),[]);
idxProx = idxProx(centerIndexX(1):centerIndexX(2),centerIndexY(1):centerIndexY(2),:);
idxProx = reshape(idxProx,[],size(idxWin,2));

data = [];
labels = [];

img = reshape(img,[],size(img,3));

for i = 1:size(idxWin,2)
    x = img(idxWin(:,i),:);
    x = reshape(x,sizeWindow(1),sizeWindow(2),[]);
    y = proximity(idxProx(:,i));
    y = reshape(y,sizePredictionPatch(1),sizePredictionPatch(2),[]);
%     y = y(centerIndexX(1):centerIndexX(2),centerIndexY(1):centerIndexY(2));
    data = cat(4,data,x);
    labels = cat(4,labels,y);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform Clustering
dataMat = reshape(data,size(data,1)*size(data,2)*size(data,3),size(data,4));
dataMat = permute(dataMat,[2,1]);
if size(data,4) < 1000
    opts.p = size(data,4);
end
clusterIdx = LSC(double(dataMat),round(sqrt(size(dataMat,2))), opts);
clear dataMat;
tbl = tabulate(clusterIdx);
sampleSize = round(median(tbl(:,2)));

tempData = [];
tempLabels = [];
for iIdx = 1:size(tbl(:,1),1)
    if tbl(iIdx,2) > sampleSize
        r = randperm(tbl(iIdx,2));
        pos = find(clusterIdx == tbl(iIdx,1));
        selectedIdx = pos(r(1:sampleSize));
        tempData = cat(4,tempData,data(:,:,:,selectedIdx));
        tempLabels = cat(4,tempLabels,labels(:,:,:,selectedIdx));
    else
        pos = find(clusterIdx == tbl(iIdx,1));
        selectedIdx = pos(1:tbl(iIdx,2));
        tempData = cat(4,tempData,data(:,:,:,selectedIdx));
        tempLabels = cat(4,tempLabels,labels(:,:,:,selectedIdx));
    end
end
data = tempData;
labels = tempLabels;
clear tempData;
clear tempLabels;

%%%%%%%%%%%%%%%%%%%%%%%%%%%

end


