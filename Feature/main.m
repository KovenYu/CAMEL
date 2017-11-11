%% settings

gpu = 1;
% Set gpu to a specific number.
% Typically, MATLAB lists gpu from 1 to N (the number of your gpus).
% If you don't have a gpu, we are sorry to tell you that our code simply 
% does not work due to some unknown internal problems.
prepareGPU(gpu);
imgFolder = 'DEMO_imgs/';
imgDir = dir([imgFolder, '*.png']);
modelDir = 'pretrained_model.mat';
model = load(modelDir);
run('./matconvnet-1.0-beta23/matlab/vl_setupnn.m')
net = dagnn.DagNN.loadobj(model.net);

%% load images

data = zeros(144, 56, 3, 0, 'single');
for i = 1:numel(imgDir)
    name = imgDir(i).name;
    data(:, :, :, i) = jstl_imread([imgFolder, name]);
end
imdb.images.data = data;

%% extract features

idx = net.getVarIndex('pool_final');
net.vars(idx).precious = true;
n = size(imdb.images.data, 4);
feaSize = 64;
images = imdb.images.data;
net.mode = 'test';
feature = zeros(feaSize, n, 'single');
net.move('gpu');

for i = 1:100:n
    upper = min(i+99, n);
    image = images(:, :, :, i:upper);
    if strcmp(net.device, 'gpu')
        image = gpuArray(image);
    end
    net.eval({'image', image});
    feature(:, i:upper) = gather(net.vars(idx).value);
end

disp('feature extraction done (in the variable "feature")');