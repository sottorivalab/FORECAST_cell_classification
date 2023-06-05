function [data,labels] = ExtractPatch_train(sizeWindow,sizePredictionPatch,stride,rgb,proximity)
% ExtractPatch extracts patch from an input rgb image and its corresponding
% ground truth image
% Inputs:
%   sizeWindow: [nrow,ncol] of a patch
%   sizePredictionPatch : [nrow,ncol] of a predicted patch
%                         Note that sizePredictionPatch should be smaller
%                         than or equal to the size of sizeWindow
%   stride: [step row, step col]
%   rgb: an rgb image
%   proximity: ground truth proximity map
% Outputs:
%   data: a 4D matrix. The first two dimensions are the spatial dimension.
%         The 3rd D is the image feature. The 4th D contain each individual patch.
%   labels: a 3D matrix. The first two dimensions are the spatial
%           dimension. The third D corresponds to an individual patch.

% Korsuk Sirinukunwattana
% BIAlab, Department of Computer Science
% 2015 - 4 - 18  include stride
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sizeImage = [size(rgb,1),size(rgb,2)];

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

rgb = reshape(rgb,[],size(rgb,3));
proximity = reshape(proximity,[],size(proximity,3));

for i = 1:size(idxWin,2)
    x = rgb(idxWin(:,i),:);
    x = reshape(x,sizeWindow(1),sizeWindow(2),[]);
    y = proximity(idxProx(:,i), :);
    y = reshape(y,sizePredictionPatch(1),sizePredictionPatch(2),[]);
%     y = y(centerIndexX(1):centerIndexX(2),centerIndexY(1):centerIndexY(2));
    data = cat(4,data,x);
    labels = cat(4,labels,y);
end

end


