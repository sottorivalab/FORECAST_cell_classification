function AnnotatedImage = AnnotateDetections(CSVPath, TilePath, LabelMap, AnnotationSize)
    AnnotatedImage = im2double(imread(TilePath)).^0.3;

    fid = fopen(CSVPath);
    
    if fid ~= -1
        CSV=textscan(fid, '%s %s %s\n', 'Delimiter', ',', 'EndOfLine', '\n');
        fclose(fid);
            
        labels = CSV{1}(2:end);
            
        X = cellfun(@str2num, CSV{2}(2:end));
        Y = cellfun(@str2num, CSV{3}(2:end));
        %unique(labels)
        %keys(LabelMap)
        cellTypes = sort(intersect(unique(labels), keys(LabelMap)));
        cellTypes = cellTypes(end:-1:1);
    
    
        for i=1:length(cellTypes)
            colour = LabelMap(cellTypes{i});
        
            if ~isempty(colour)
                isCellType = strcmp(labels, cellTypes{i});
                AnnotatedImage = annotate_image_with_class(AnnotatedImage, [X(isCellType), Y(isCellType)], colour, AnnotationSize);
            end
        end
    end
end

