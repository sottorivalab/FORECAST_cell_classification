function M = GenerateDetectionGTProximity(centroids,r,alpha,imgSize)
% GenerateDetectionGTProximity generates a ground truth proximity mask 
%
% Inputs:
%   centroids: nuclear coordinates
%   r: a radius of a weight mask
%   alpha: decay rate of a weight mask
% Outputs:
%   M: a proximity mask
%
% Korsuk Sirinukunwattana
% BIAlab, Department of Computer Science, University of Warwick
% 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Generate a weight mask and a binary mask
[weightMask,binaryMask] = GenerateCircleMask(r,alpha);

centroids = round(centroids);

M = zeros(imgSize,'single');
count = zeros(imgSize,'single');

for i = 1:size(centroids,1)
    row = centroids(i,2)-r:centroids(i,2)+r;
    col = centroids(i,1)-r:centroids(i,1)+r;

    if min(row) < 1 || max(row) > size(M,1) ||...
            min(col) < 1 || max(col) > size(M,2)
        continue;
    end
    
    M(row,col) = M(row,col) + weightMask;
    count(row,col) = count(row,col) + binaryMask;
end
M(count>0) = M(count>0)./count(count>0);

end

function [weightMask,binaryMask] = GenerateCircleMask(r,alpha)

[X,Y] = meshgrid(-r:r,-r:r);
D = sqrt(X.^2 + Y.^2);
weightMask = 1./(1+alpha*D.^2);
weightMask(D > r) = 0;
binaryMask = D <= r;

end


