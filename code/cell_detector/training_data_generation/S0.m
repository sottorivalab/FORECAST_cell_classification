function S0(AnnotationsFolder, ImageFolder, PositiveFolder, NegativeFolder, NegativeLabels)
    saveFolder = {PositiveFolder, NegativeFolder};
    dirs = dir(AnnotationsFolder);
    dirs = dirs(~ismember({dirs.name}, {'.', '..'}));

    for d = 1:length(dirs)
        CSVFolder = fullfile(AnnotationsFolder, dirs(d).name);
        CurrImageFolder = fullfile(ImageFolder, dirs(d).name);    
        files = dir(fullfile(CSVFolder, 'Da*.csv'));
        
        for i = 1:length(files)
            A = importdata(fullfile(files(i).folder, files(i).name));
            detections = A.data;
            labels = A.textdata(2:end, 1);
            im = imread(fullfile(CurrImageFolder, [files(i).name(1:end-4), '.jpg']));
            
            if ~isempty(detections)
                labels(detections(:, 1)<=0 | detections(:, 2)<=0,:) = [];
                detections(detections(:, 1)<=0 | detections(:, 2)<=0,:) = [];
                negatives = ismember(labels, NegativeLabels);
                for j=1:2
                    if j==1
                        selectedCells = ~negatives;
                    elseif j==2
                        selectedCells = negatives;
                    end
                    
                    Z = zeros([size(im,1), size(im,2)]);
                    linearInd = sub2ind(size(Z), floor(detections(selectedCells,2)), floor(detections(selectedCells,1)));
                    Z(linearInd) = 1;
                    [data, labels] = ExtractPatch_train([200,200],[200,200],[180,180],im, Z);           
            
                    for n = 1:size(data,4)
                        if sum(sum(labels(:,:,:,n)))
                            filename = sprintf('%s_%s_%d',dirs(d).name, files(i).name(1:end-4),n);
                            mkdir(fullfile(saveFolder{j}, filename));
                            [r,c] = find(labels(:,:,:,n));
                            detection = [c, r];
                            
                            imwrite(data(:,:,:,n), fullfile(saveFolder{j}, filename, [filename, '.bmp']));
                            save(fullfile(saveFolder{j}, filename, [filename, '_detection.mat']),'detection');
                        end
                    end
                end
            end
        end
    end
end
