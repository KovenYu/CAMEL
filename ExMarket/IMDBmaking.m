%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This m-file construct the ExMarket dataset and save it into
% 'ExMarket_IMDB.mat',
% where
% images.data is the original images,
% images.labels is the corresponding labels,
% images.idxViews is the corresponding index of camera views,
% images.set indicates the set splitting: 1.train 2.test(gallery) 3.query(probe)
%
% IMPORTANT NOTE: the size of original images from Market is 128-by-64,
% while the size of those from MARS is 256-by-128. This may be a potential
% problem when extracting feature, so we recommand that the user first
% resize the images.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear, clc
data = cell(1, 0);
idxViews = zeros(1, 0, 'single');
labels = zeros(1, 0, 'single');
set =zeros(1, 0, 'uint8');

%% Market

folder = {'train', 'test', 'query'};
for i = 1:numel(folder)
    path = ['/Market-1501-v15.09.15/', folder{i}];
    % This is a Linux path which might need to be updated according to your machine.
    directory = dir(path);
    idx = arrayfun(@(x)length(x.name)<5||~strcmp(x.name(5), '_'), directory);
    directory = directory(~idx);
    n = numel(directory);
    data_t = cell(1,n);
    labels_t = zeros(1, n, 'single');
    idxViews_t = zeros(1, n, 'single');
    set_t = ones(1, n, 'uint8');
    for j = 1:n
        filename = directory(j).name;
        labels_t(j) = single(str2double(filename(1:4)));
        idxViews_t(j) = single(str2double(filename(7)));
        filepath = [path, '/', filename];
        data_t{j} = imread(filepath);
        set_t(j) = i;
        if mod(j, 1000) == 0
            fprintf('j == %d\n', j)
        end
    end
    data = cat(2, data, data_t);
    labels = cat(2, labels, labels_t);
    set = cat(2, set, set_t);
    idxViews = cat(2, idxViews, idxViews_t);
end

%% MARS

data_t = cell(1, 250000);
idxViews_t = zeros(1, 250000, 'single');
labels_t = -100*ones(1, 250000, 'single');
set_t = zeros(1, 250000, 'uint8');

folderPath = 'MARS-v160809(large video re-id)/bbox_test/';
% This is a Linux path which might need to be updated according to your machine.
mainDir = dir(folderPath);
t = 0;
for i = 5:length(mainDir)
    label = str2double(mainDir(i).name);
    fprintf('test set, i = %d/%d \r', i, length(mainDir));
    subDir = dir([folderPath, mainDir(i).name]);
    for j = 3:5:length(subDir)
        t = t+1;
        filename = [folderPath, mainDir(i).name, '/', subDir(j).name];
        name = subDir(j).name;
        cam = str2double(name(6));
        data_t{t} = imread(filename);
        labels_t(t) = label;
        idxViews_t(t) = cam;
    end
end

folderPath = 'MARS-v160809(large video re-id)/bbox_train/';
% This is a Linux path which might need to be updated according to your machine.
mainDir = dir(folderPath);
for i = 3:length(mainDir)
    label = str2double(mainDir(i).name);
    fprintf('training set, i = %d/%d \r', i, length(mainDir));
    subDir = dir([folderPath, mainDir(i).name]);
    for j = 3:5:length(subDir)
        t = t+1;
        filename = [folderPath, mainDir(i).name, '/', subDir(j).name];
        name = subDir(j).name;
        cam = str2double(name(6));
        data_t{t} = imread(filename);
        labels_t(t) = label;
        idxViews_t(t) = cam;
    end
end

idTrain = unique(labels(set == 1));
idTest = unique(labels(set == 2));
for i = 1:length(idTrain)
    idHere = idTrain(i);
    idx = labels_t == idHere;
    set_t(idx) = 1;
end
for i = 2:length(idTest)
    idHere = idTest(i);
    idx = labels_t == idHere;
    set_t(idx) = 2;
end

data_t(cellfun(@isempty, data_t)) = [];
idxViews_t(idxViews_t == 0) = [];
labels_t(labels_t == -100) = [];
set_t(set_t == 0) = [];


%% Combine them

images.data = cat(2, data, data_t);
images.labels = cat(2, labels, labels_t);
images.idxViews = cat(2, idxViews, idxViews_t);
images.set = cat(2, set, set_t);

save('ExMarket_IMDB.mat', 'images', '-v7.3')