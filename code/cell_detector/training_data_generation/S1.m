function S1(trainingDatasetPath, savematpath, fileprefix)
    if (nargin < 3)
        fileprefix = [];
    end
    
    features = {'h','rgb'};         % choice for an image type
    celltype = 'detection';       
    windowSize = [31,31];           % size of a patch
    predictPatchSize = [13,13];     % prediction patch size
    stride = [8,8];                 % stride
    valPercentage = 10;             % percentage for validation
    
    DataSets = {'Training', 'Validation'};
    
    mkdir(savematpath);
    
    for i=1:length(DataSets)
        mkdir(fullfile(savematpath, DataSets{i}));
    end
    
    %% training image folders
    folders = dir(trainingDatasetPath);
    isub = [folders(:).isdir];      %# returns logical vector
    folders = {folders(isub).name}';
    folders(ismember(folders,{'.','..'})) = [];

    %% Create imdb here
    trainFolders = folders;
    imagesPath = cell(length(trainFolders),1);

    for j = 1:length(trainFolders)
        imagesPath{j} = fullfile(trainingDatasetPath,trainFolders{j});
    end
    
    [trainFeats, trainHandmark, trainIdx] = GenerateDetectionDataset(imagesPath,celltype,valPercentage,features);

    iter = 1;
    
    for i = 1:size(trainFeats,4)
        [dataSet,labelsSet] = ExtractPatch_train(windowSize,predictPatchSize,stride,trainFeats(:,:,:,i),trainHandmark(:,:,:,i));

        dataSet = dataSet./255;
        
        set = 2-double(trainIdx(i));

        for j = 1:size(dataSet,4)
            data = dataSet(:,:,:,j);
            labels = labelsSet(:,:,:,j);
            
            if any(labels(:))
                save(fullfile(savematpath,DataSets{set},sprintf('%s%d.mat',fileprefix,iter)), 'data', 'labels', 'set');
                iter = iter+1;
            end
        end
    end
end

function [allImages, allProximity,trainIdx] = GenerateDetectionDataset(imagesPath,cellType,valPercentage,features)
    allImages = [];
    allProximity = [];
    group = zeros(length(imagesPath),1);

    % calculate features and proximity
    for iImage = 1:length(imagesPath)
        file = dir(fullfile(imagesPath{iImage},'*.bmp'));
        fprintf('Processing %s ... %2.2f %% complete \n', file.name, 100*iImage/length(imagesPath));
        I = imread(fullfile(imagesPath{iImage},file.name));  % read an RGB image
        I = Retinex(I);     % adjust an image  
    
        % feature
        feat = [];
        for iFeatures = 1:length(features)
            switch features{iFeatures}
                case 'rgb'
                    feat = cat(3,feat,single(I));
                case 'lab'
                    feat = cat(3,feat,single(rgb2lab(I)));
                case 'h'
                    doubleRGB = RemoveArtefact(I);
              
                    colourMat = EstUsingSCD(doubleRGB);
                    [ DCh ] = Deconvolve( I, colourMat );
                    [ H ] = PseudoColourStains( DCh, colourMat );
                    H = rgb2gray(H);
                    feat = cat(3,feat,single(H));
                case 'e'
                    doubleRGB = RemoveArtefact(I);
                    colourMat = EstUsingSCD(doubleRGB);
                    [ DCh ] = Deconvolve( I, colourMat );
                    [ ~, E ] = PseudoColourStains( DCh, colourMat );
                    E = rgb2gray(E);
                    feat = cat(3,feat,single(E));
                case 'he'
                    doubleRGB = RemoveArtefact(I);
                    colourMat = EstUsingSCD(doubleRGB);
                    [ DCh ] = Deconvolve( I, colourMat );
                    [ H, E ] = PseudoColourStains( DCh, colourMat );
                    H = rgb2gray(H);
                    E = rgb2gray(E);
                    feat = cat(3,feat,single(H),single(E));
                case 'br'
                    BR = BlueRatioImage(I);
                    feat = cat(3,feat,single(BR));
                case 'grey'
                    grey = rgb2gray(I);
                    feat = cat(3,feat,single(grey));
            end
        end   
        allImages = cat(4,allImages,feat);
    
        % proximity
        if exist(fullfile(imagesPath{iImage},[file.name(1:end-4),'_',cellType,'.mat']),'file')
            J = load(fullfile(imagesPath{iImage},[file.name(1:end-4),'_',cellType,'.mat']));  % read a ground truth image
            J = J.detection;
            group(iImage) = 1;
        else
            J = [];
        end
            
        alpha = 0.5;
        M = GenerateDetectionGTProximity(J,5,alpha,[size(feat,1),size(feat,2)]);           % r = 4, 
        allProximity = cat(4,allProximity,M);
    end
    
    cv = cvpartition(group,'HoldOut',valPercentage/100);
    trainIdx = training(cv,1);
end
