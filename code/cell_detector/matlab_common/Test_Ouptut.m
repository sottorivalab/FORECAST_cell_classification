csv_file_name = 'E:\TracerX_Lung\results\100xsamples\Region_Specific_HE\detection\csv\LTX022_N.ndpi\Da9.csv';
colorcodes = readtable('HE_Fib_Lym_Tum_Others.csv');
A = readtable(csv_file_name);
strength = 2;
image_path_full = 'E:\TracerX_Lung\data\100xSamples\Region_Specific_HE\data\cws\LTX022_N.ndpi\Da9.jpg';
image = imread(image_path_full);
detection = [A.V2, A.V3];
mat = load('E:\TracerX_Lung\results\100xsamples\Region_Specific_HE\classification\classification-20170821\mat\LTX022_N.ndpi\Da9.mat');
if isfield(mat, 'mat')
    mat = mat.mat;
end
cell_ids = mat.cell_ids;
C = unique(cell_ids);
class = zeros(length(C),1);

for j = 1:length(C)
    classes_9 = mat.output(mat.cell_ids==C(j));
    class(j) = mode(classes_9);
    if length(unique(classes_9))>2
        class(j) = 4;
    end
    if sum(classes_9==3)<6 && class(j)~=4
        classes_9(classes_9==3) = [];
        class(j) = mode(classes_9);
    end
end
for c = 1:height(colorcodes)
    image = annotate_image_with_class(image, detection(class==c,:), ...
        hex2rgb(colorcodes.color{c}), strength);
end
figure, imshow(image);
A.V1 = colorcodes.class(class);