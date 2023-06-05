function WriteAnnotations(ImageNamePattern, DetectionPath, TilePath, OutPath, ColourCodeFile, AnnotationSize, OverWrite)
    if nargin < 6
        AnnotationSize = 6;
    end
    
    if nargin < 7
        OverWrite = false;
    end

    fid = fopen(ColourCodeFile);
    colourCodes=textscan(fid, '%s %s %*[^\n]', 'Delimiter', ' ');
    fclose(fid);
    
    colourCodes = cat(2, colourCodes{:});

    labelNames = colourCodes(:, 2);
    colourCodes = colourCodes(:, 1);
    
    parfor i = 1:length(colourCodes)
        colourCode = colourCodes{i};

        if strcmp(labelNames{i}, 'unk')
            labelColours{i} = [];
        else
            labelColours{i} = (hex2dec({colourCode(1:2), colourCode(3:4), colourCode(5:6)})')./255;
        end
    end
    
    labelMap = containers.Map(labelNames, labelColours);

    TilePath = dir(fullfile(TilePath, '/'));
    TilePath = TilePath(1).folder;
    
    tileFiles = dir(fullfile(TilePath, ImageNamePattern, 'Da*.jpg'));
    csvFiles = dir(fullfile(DetectionPath, ImageNamePattern, 'Da*.csv'));

    if isempty(csvFiles)
        error('No CSV Files Found!');
    else
        outFolders = unique({tileFiles.folder});
    
        for i = 1:length(outFolders)
            mkdir(fullfile(OutPath, outFolders{i}((length(TilePath)+1):end)));
        end

        parfor i = 1:length(tileFiles)
            [~, TileName, ~] = fileparts(tileFiles(i).name);
            currTileFile = fullfile(tileFiles(i).folder, tileFiles(i).name);
            
            currCSVFile = fullfile(DetectionPath, tileFiles(i).folder((length(TilePath)+1):end), [TileName '.csv']);
            currOutTileFile = fullfile(OutPath, tileFiles(i).folder((length(TilePath)+1):end), [TileName '.jpg']);

            if OverWrite || ~(exist(currOutTileFile, 'file'))
                fprintf('Annotating tile for: %s\n', currTileFile);
                annotatedImage = AnnotateDetections(currCSVFile, currTileFile, labelMap, AnnotationSize);
                imwrite(annotatedImage, currOutTileFile);
            else
                fprintf('Annotated tile already exists for: %s, skipping.\n', currOutTileFile);
            end
        end
    end 
end
