function GenerateImagePatchesFromCSVs(CSVPath, TilePath, OutPath, nPatches, PatchSize, CellLabels, nAngles, MaxShift)
    annotations = table();

    csvFolders = dir(CSVPath);
    csvFolders = csvFolders(~ismember({csvFolders.name},{'.','..'}));
    
    for i=1:length(csvFolders)    
        [~, ~, ext] = fileparts(csvFolders(i).name);

        if strcmpi(ext, '.czi')
            readPositions = [1 6];
        else
            readPositions = [7 20];
        end

        fid = fopen(fullfile(TilePath, csvFolders(i).name, '/FinalScan.ini'), 'r');
        tileDims = textscan(fid, '%s', 2, 'delimiter', '\n', 'headerlines', readPositions(1));
        tileDims = split(tileDims{:}, '=');
        tileDims = str2double(tileDims(:, 2));

        imageDims = textscan(fid, '%s', 2, 'delimiter', '\n', 'headerlines', readPositions(2));
        imageDims = split(imageDims{:}, '=');
        imageDims = str2double(imageDims(:, 2));

        fclose(fid);

        tileGridDims = ceil(imageDims./tileDims);
    
        csvFiles = dir(fullfile(csvFolders(i).folder, csvFolders(i).name, 'Da*.csv'));
        
        for j=1:length(csvFiles)
            [~, fName, ~] = fileparts(csvFiles(j).name);
            T = readtable(fullfile(csvFiles(j).folder, csvFiles(j).name));
            T.V4 = repmat({csvFolders(i).name}, height(T), 1);
            T.V5 = repmat({fName}, height(T), 1);
            T.V6 = repmat({tileGridDims(1)}, height(T), 1);
            T.V7 = repmat({tileGridDims(2)}, height(T), 1);
            annotations = [annotations; T];
        end
    end
    
    removeAnnos = ~ismember(annotations.V1, CellLabels);
    
    annotations(removeAnnos, :) = [];
    
    angles = linspace(0, 360, nAngles+1);
    angles(end) = [];
    [translationsX, translationsY] = meshgrid(-MaxShift:MaxShift, -MaxShift:MaxShift);

    for i=1:length(CellLabels)
        typeAnnos = annotations(strcmp(annotations.V1, CellLabels{i}), :);
        
        if ~isempty(typeAnnos)
            mkdir(fullfile(OutPath, CellLabels{i}));
            nCombos = height(typeAnnos)*nAngles*(2*MaxShift + 1).^2;

            selections = randperm(nCombos, nPatches)-1;

            annoIDXs = mod(selections, height(typeAnnos));
            selections = (selections-annoIDXs)./height(typeAnnos);
            annoAngleIDXs = mod(selections, nAngles);
            selections = (selections-annoAngleIDXs)./nAngles;
            annoTranIDXs = selections;

            selectedAnnos = arrayfun(@(x) typeAnnos(x+1, :), annoIDXs, 'UniformOutput', false);
            annoAngles = angles(annoAngleIDXs+1);
            annoTX = translationsX(annoTranIDXs+1);
            annoTY = translationsY(annoTranIDXs+1);

            parfor j=1:length(selections)
                anno = selectedAnnos{j};
                folderName = anno.V4;
                tileName = anno.V5;
                tileGridDims = [anno.V6{:} anno.V7{:}];
                
                tileIDX = str2double(tileName{:}(3:end));
                
                xCentre = anno.V2 + annoTX(j);
                yCentre = anno.V3 + annoTY(j);
                
                xRange = xCentre + [-1 1].*((PatchSize(1)-1)/2);
                yRange = yCentre + [-1 1].*((PatchSize(2)-1)/2);
                
                tileArea = [xRange(1) yRange(1); xRange(2) yRange(1); xRange(2) yRange(2); xRange(1) yRange(2)];
                Mat = [1 0 0; 0 1 0; -xCentre -yCentre 1]*[cosd(annoAngles(j)) sind(annoAngles(j)) 0; -sind(annoAngles(j)) cosd(annoAngles(j)) 0; 0 0 1]*[1 0 0; 0 1 0; xCentre yCentre 1];
                
                rTileArea = [tileArea ones(size(tileArea, 1), 1)]*Mat;
                rTileArea = [floor(min(rTileArea, [], 1)); ceil(max(rTileArea, [], 1))];
                
                tile = imread(fullfile(TilePath, folderName{:}, [tileName{:} '.jpg']));
                
                srTileArea = rTileArea;

                if rTileArea(2, 1) > size(tile, 2)
                    if mod(tileIDX, tileGridDims(:, 1)) < tileGridDims(:, 1)-1
                        rightTile = imread(fullfile(TilePath, folderName{:}, ['Da' num2str(tileIDX+1) '.jpg']));
                        rightTile = rightTile(:, 1:rTileArea(2, 1)-size(tile, 2), :);
                        tile = [tile rightTile];
                    else
                        padSize = rTileArea(2, 1)-size(tile, 2);
                        tile = padarray(tile, [0 padSize], 'symmetric', 'post');
                    end
                end
                
                if rTileArea(1, 1) < 1
                    if mod(tileIDX, tileGridDims(:, 1)) > 0
                        leftTile = imread(fullfile(TilePath, folderName{:}, ['Da' num2str(tileIDX-1) '.jpg']));
                        leftTile = leftTile(:, end+rTileArea(1, 1):end, :);
                        tile = [leftTile tile];
                        
                        srTileArea(:, 1) = srTileArea(:, 1) + size(leftTile, 2);
                    else
                        padSize = 1-rTileArea(1, 1);
                        tile = padarray(tile, [0 padSize], 'symmetric', 'pre');
                        
                        srTileArea(:, 1) = srTileArea(:, 1) + padSize;
                    end
                end
                
                if rTileArea(2, 2) > size(tile, 1)
                    if floor(tileIDX/tileGridDims(:, 1)) < tileGridDims(:, 2)-1
                        bottomTile = imread(fullfile(TilePath, folderName{:}, ['Da' num2str(tileIDX+tileGridDims(:, 1)) '.jpg']));
                        bottomTile = bottomTile(1:rTileArea(2, 2)-size(tile, 1), :, :);
                        
                        if rTileArea(2, 1) > size(bottomTile, 2)
                            if mod(tileIDX, tileGridDims(:, 1)) < tileGridDims(:, 1)-1
                                rightTile = imread(fullfile(TilePath, folderName{:}, ['Da' num2str(tileIDX+tileGridDims(:, 1)+1) '.jpg']));
                                rightTile = rightTile(1:rTileArea(2, 2)-size(tile, 1), 1:rTileArea(2, 1)-size(bottomTile, 2), :);
                                bottomTile = [bottomTile rightTile];
                            else
                                padSize = rTileArea(2, 1)-size(bottomTile, 2);
                                bottomTile = padarray(bottomTile, [0 padSize], 'symmetric', 'post');
                            end
                        end
                        
                        if rTileArea(1, 1) < 1
                            if mod(tileIDX, tileGridDims(:, 1)) > 0
                                leftTile = imread(fullfile(TilePath, folderName{:}, ['Da' num2str(tileIDX+tileGridDims(:, 1)-1) '.jpg']));
                                leftTile = leftTile(1:rTileArea(2, 2)-size(tile, 1), end+rTileArea(1, 1):end, :);
                                bottomTile = [leftTile bottomTile];
                            else
                                padSize = 1-rTileArea(1, 1);
                                bottomTile = padarray(bottomTile, [0 padSize], 'symmetric', 'pre');
                            end
                        end
                        
                        tile = [tile; bottomTile];
                    else
                        padSize = rTileArea(2, 2)-size(tile, 1);
                        tile = padarray(tile, [padSize 0], 'symmetric', 'post');
                    end
                end
                
                if rTileArea(1, 2) < 1
                    if floor(tileIDX/tileGridDims(:, 1)) > 0
                        topTile = imread(fullfile(TilePath, folderName{:}, ['Da' num2str(tileIDX-tileGridDims(:, 1)) '.jpg']));
                        topTile = topTile(end+rTileArea(1, 2):end, :, :);
                        
                        if rTileArea(2, 1) > size(topTile, 2)
                            if mod(tileIDX, tileGridDims(:, 1)) < tileGridDims(:, 1)-1
                                rightTile = imread(fullfile(TilePath, folderName{:}, ['Da' num2str(tileIDX-tileGridDims(:, 1)+1) '.jpg']));
                                rightTile = rightTile(end+rTileArea(1, 2):end, 1:rTileArea(2, 1)-size(topTile, 2), :);
                                topTile = [topTile rightTile];
                            else
                                padSize = rTileArea(2, 1)-size(topTile, 2);
                                topTile = padarray(topTile, [0 padSize], 'symmetric', 'post');
                            end
                        end
                        
                        if rTileArea(1, 1) < 1
                            if mod(tileIDX, tileGridDims(:, 1)) > 0
                                leftTile = imread(fullfile(TilePath, folderName{:}, ['Da' num2str(tileIDX-tileGridDims(:, 1)-1) '.jpg']));
                                leftTile = leftTile(end+rTileArea(1, 2):end, end+rTileArea(1, 1):end, :);
                                topTile = [leftTile topTile];
                            else
                                padSize = 1-rTileArea(1, 1);
                                topTile = padarray(topTile, [0 padSize], 'symmetric', 'pre');
                            end
                        end
                        
                        tile = [topTile; tile];
                        
                        srTileArea(:, 2) = srTileArea(:, 2) + size(topTile, 1);
                    else
                        padSize = 1-rTileArea(1, 2);
                        tile = padarray(tile, [padSize 0], 'symmetric', 'pre');
                        
                        srTileArea(:, 2) = srTileArea(:, 2) + padSize;
                    end
                end
                
                tPatch = tile(srTileArea(1, 2):srTileArea(2, 2), srTileArea(1, 1):srTileArea(2, 1), :);

                worldCoordRef = imref2d(size(tPatch), rTileArea(1:2, 1)', rTileArea(1:2, 2)');
                localCoordRef = imref2d(PatchSize, xRange, yRange);
                
                patch = imwarp(tPatch, worldCoordRef, invert(affine2d(Mat)), 'OutputView', localCoordRef);

                imwrite(patch, fullfile(OutPath, CellLabels{i}, [folderName{:} '_' tileName{:} '_' num2str(anno.V2) '_' num2str(anno.V3) '_' num2str(round(annoAngles(j))) '_' num2str(annoTX(j)) '_' num2str(annoTY(j)) '.png']));
            end
        end
    end
end
