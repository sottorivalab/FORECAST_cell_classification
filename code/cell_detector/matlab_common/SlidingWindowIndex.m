function idxWin = SlidingWindowIndex(sizeImage,sizeWindow,stride)
% SlidingWindowIndex generates linear indices for a sliding window 
% Inputs:
%   sizeImage: a 2-by-1 vector of dimensions of an image. 
%              sizeImage(1) = row dimension
%              sizeImage(2) = col dimension
%   sizeWindow: a 2-by-1 vector of dimensions of a sliding window.
%              sizeWindow(1) = row dimension
%              sizeWindow(2) = col dimension
%   stride: a 2-by-1 vector of steps between cosecutive sliding windows.
%              stride(1) = step size in row direction
%              stride(2) = step size in col direction
%
% Outputs
%   idxWin : a matrix whose column is a linear index for each sliding
%            window
%
% Korsuk Sirinukunwattana
% BIAlab, Department of Computer Science, University of Warwick
% 2015-4-13
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[II,JJ]=ndgrid(1:sizeWindow(1), 1:sizeWindow(2));
idxWin=sub2ind(sizeImage,II(:),JJ(:));

strideRow = 0:stride(1):sizeImage(1)-sizeWindow(1);
strideCol = reshape(0:stride(2):sizeImage(2)-sizeWindow(2),1,1,[]);

idxWin=bsxfun(@plus,idxWin,strideRow);
idxWin=bsxfun(@plus,idxWin,sizeImage(1)*strideCol);

idxWin=reshape(idxWin,prod(sizeWindow),[]);

end