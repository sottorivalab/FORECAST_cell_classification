function MergeCSVs(inFolder, tileFolder, outCSV)
    if nargin < 2
        tileFolder = inFolder;
    end
    
    if nargin < 3
        outCSV = fullfile(inFolder, 'AllCells.csv');
    end
    
    fScanText = fileread(fullfile(tileFolder, '/FinalScan.ini'));
    
    tWidth = regexp(fScanText, '(iImageWidth=)(\d*)', 'tokens');
    tHeight = regexp(fScanText, '(iImageHeight=)(\d*)', 'tokens');
    TileDims = [str2double(tWidth{1}{2}), str2double(tHeight{1}{2})];
    
    iWidth = regexp(fScanText, '(iWidth=)(\d*)', 'tokens');
    iHeight = regexp(fScanText, '(iHeight=)(\d*)', 'tokens');
    ImageDims = [str2double(iWidth{1}{2}), str2double(iHeight{1}{2})];
    
    TileGridDims = ceil(ImageDims./TileDims);
    
    csvs = dir(fullfile(inFolder, 'Da*.csv'));
    cells = [];
    
    for i=1:length(csvs)
        TileTable = readtable(fullfile(csvs(i).folder, csvs(i).name), 'ReadVariableNames', true);
        idx = str2double(csvs(i).name(3:end-4));
        
        TileX = TileDims(1)*mod(idx, TileGridDims(1));
        TileY = TileDims(2)*floor(idx/TileGridDims(1));

        TileTable.V2 = TileTable.V2+TileX;
        TileTable.V3 = TileTable.V3+TileY;
        
        cells = [cells; TileTable];
    end
    writetable(cells, outCSV);
end

