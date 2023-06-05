function process_csv_for_classification_training( curr_wsi, curr_csv, curr_norm_methods, im, csv_table, curr_save_folder, network_parameters )
%process_csv_for_classification_training Summary of this function goes here
%   Detailed explanation goes here

celltypeMap = containers.Map(network_parameters.cell_types, 1:length(network_parameters.cell_types));
validLabels = cellfun(@(x) isKey(celltypeMap, x), csv_table.V1);

if any(~validLabels)
    badLabels = unique(csv_table.V1(~validLabels));
    warning('The following cell labels are not recognised:\n%s', strjoin(badLabels));
end

csv_table(~validLabels, :) = [];

labelIndices = cellfun(@(x) celltypeMap(x), csv_table.V1);

angles = linspace(0, 360, network_parameters.nangles+1);
angles(end) = [];
[translationsX, translationsY] = meshgrid(-network_parameters.radiusFromCentroid:network_parameters.radiusFromCentroid, -network_parameters.radiusFromCentroid:network_parameters.radiusFromCentroid);

for norm = 1:length(curr_norm_methods)
    im = normalise_image(im, curr_norm_methods(norm));
end

for angle = 1:numel(angles)
    [im_rot, csv_table_rot] = rotate_annotations(im, csv_table, angles(angle));
    feat = get_feat(im_rot, network_parameters.features);

    for translation = 1:numel(translationsX)
        xCoords = csv_table_rot.V2+translationsX(translation);
        yCoords = csv_table_rot.V3+translationsY(translation);

        params = [angles(angle) translationsX(translation) translationsY(translation)];

        for t = 1:height(csv_table)
            labels = labelIndices(t);
            centroids = [xCoords(t), yCoords(t)];
            save_patches([t params], curr_wsi, curr_csv, feat, labels, centroids, curr_save_folder, network_parameters);
        end
    end
end

end

function out_im = normalise_image(im, normalisation)

switch normalisation
    case 0
        I = im2uint8(im);
    case 1
        I = im2uint8(Retinex(im));         % adjust using Retinex
    case 2
        I = Retinex(im);         % adjust using Retinex
        TargetImage = imread('Target.png');
        [ I ] = im2uint8(NormReinhard( I, TargetImage));
    case 3
        TargetImage = imread('Target.png');
        [ I ] = im2uint8(NormReinhard( im, TargetImage));
end

out_im = I;    
end

function feat = get_feat(I, features)

feat = [];
    for iFeatures = 1:length(features)
        switch features{iFeatures}
            case 'rgb'
                feat = cat(3,feat,single(I));
            case 'lab'
                feat = cat(3,feat,single(rgb2lab(I)));
            case 'h'
                doubleRGB = RemoveArtefact(I);
                colourMat = EstUsingMacenko(doubleRGB);
                [ DCh ] = Deconvolve( I, colourMat );
                H = uint8((255*exp(-DCh(:,:,1)))-1);
                feat = cat(3,feat,single(H));
            case 'e'
                doubleRGB = RemoveArtefact(I);
                colourMat = EstUsingMacenko(doubleRGB);
                [ DCh ] = Deconvolve( I, colourMat );
                E = uint8((255*exp(-DCh(:,:,2)))-1);
                feat = cat(3,feat,single(E));
            case 'he'
                doubleRGB = RemoveArtefact(I);
                colourMat = EstUsingMacenko(doubleRGB);
                [ DCh ] = Deconvolve( I, colourMat );
                H = uint8((255*exp(-DCh(:,:,1)))-1);
                E = uint8((255*exp(-DCh(:,:,2)))-1);
                feat = cat(3,feat,single(H),single(E));
            case 'br'
                BR = BlueRatioImage(I);
                feat = cat(3,feat,single(BR));
            case 'grey'
                grey = rgb2gray(I);
                feat = cat(3,feat,single(grey));
        end
    end

end

function doubleRGB = RemoveArtefact(I)
load(fullfile('matlab','gm.mat'));

od = -log((double(I)+1)/256);       % convert RGB to OD space
od = reshape(od,[],3);  

idx = cluster(gm,od);               % remove artefact
idx = reshape(idx,size(I,1),size(I,2));
im = reshape(I,[],3);
im = im2double(im);
flag = idx == 3;
im(flag,:) = 1;

doubleRGB = reshape(im,size(I,1),size(I,2),size(I,3));

end

function save_patches(iter, curr_wsi, curr_csv, feat, labels, centroids, curr_save_folder, network_parameters) %#ok<INUSL>
I = feat;
windowSize = network_parameters.windowSize;
radiusFromCentroid = network_parameters.radiusFromCentroid;

% Preprocess an input image
padsize = [(windowSize(1)-1)/2+radiusFromCentroid,(windowSize(2)-1)/2+radiusFromCentroid];            % pad array
I = padarray(I,padsize,'symmetric');

centroids = round(centroids);
centroids(:,1) = centroids(:,1) + padsize(1);                   % relative position of centroids in the padded image
centroids(:,2) = centroids(:,2) + padsize(2);

listPixels = centroids;

for iPix = 1:size(listPixels,1)
    patchRow = listPixels(iPix,1) - (windowSize(1)-1)/2 : listPixels(iPix,1) + (windowSize(1)-1)/2;
    patchCol = listPixels(iPix,2) - (windowSize(2)-1)/2 : listPixels(iPix,2) + (windowSize(2)-1)/2;
    if min(patchRow)<1 || max(patchRow) > size(I,1) ||...
            min(patchCol) < 1 || max(patchCol)  > size(I,2)
        continue
    end
    data = I(patchRow,patchCol,:)./255; %#ok<NASGU>
    save(fullfile(curr_save_folder, sprintf('%s_%s_%d_%d_%d_%d.mat', curr_wsi, curr_csv, iter(:))), ...
        'data', 'labels');
end

end

function [im_rot, csv_table_rot] = rotate_annotations(im, csv_table, angle)
    im_size = size(im);
    
    sub_angle = 90-abs(90-mod(angle, 180));
    rot_offset = [(im_size(1)/2)*cosd(sub_angle)+(im_size(2)/2)*sind(sub_angle), (im_size(1)/2)*sind(sub_angle)+(im_size(2)/2)*cosd(sub_angle)];
    csv_table_rot = table(csv_table.V1, round((csv_table.V2-(im_size(2)/2)).*cosd(angle)-(csv_table.V3-(im_size(1)/2)).*sind(angle)+rot_offset(2)), round((csv_table.V2-(im_size(2)/2)).*sind(angle)+(csv_table.V3-(im_size(1)/2)).*cosd(angle)+rot_offset(1)), 'VariableNames', {'V1', 'V2', 'V3'});
    
    padding = ceil(abs(im_size(2:-1:1).*sind(angle).*cosd(angle)));
    im_rot = padarray(im, padding, 'symmetric');
    im_rot = imrotate(im_rot, angle);
    rot_size = size(im_rot);
    crop_offset = round((rot_size(1:2)/2)-rot_offset)+1;
    
    im_rot = im_rot(crop_offset(1):(rot_size(1)-crop_offset(1)+1),crop_offset(2):(rot_size(2)-crop_offset(2)+1), :);
end
