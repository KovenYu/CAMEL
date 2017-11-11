%% Interface of CAMEL.p
% Usage: dist = CAMEL(data4training, K, lambda, ...
%                     Gal, Prb, para)
%
% INPUT:
%
% data4training, the training data, arranged as a d-by-n matrix,
% where d is the feature dimension and n is the number of samples.
%
% K, the number of centroids.
% 
% lambda, the cross-view regularizer.
%
% Gal, a d-by-ng feature matrix,
% where d is the feature dimension and ng is the number of gallery samples.
%
% Prb, a d-by-np feature matrix,
% where d is the feature dimension and ny is the number of probe samples.
%
% para, the parameters for the dataset, should contain:
% para.numViews, the number of cameara views. (typically 2)
% para.idxViewTrain, a 1-by-n array containing the index of views of the training
% samples. From 1, 2, ..., to para.numViews.
% para.idxViewGal, a 1-by-ng array containing the index of views of the
% gallery samples.
% para.idxViewPrb, a 1-by-ng array containing the index of views of the
% probe samples.

% OUTPUT:
% dist, a nx-by-ny distance matrix.
% Note that you can put all samples in both Gal and Prb to have a full
% distance matrix.
% 
% If this code is helpful in your research, please kindly cite our work:
%
% Hong-Xing Yu, Ancong Wu, Wei-Shi Zheng,
% Cross-view Asymmetric Metric Learning for Unsupervised Person
% Re-identification,
% Proceedings of the IEEE International Conference on Computer Vision,
% 2017.
%
%                                                    Hong-Xing Yu

%% DEMO
clear
clc
warning off all
load('CUHK01_JSTL64.mat');
feature = reshape(images.data, [64, 3884]);
lambda = 1e-2;
% if you use different feature other than JSTL-64, you may want to
% set a larger lambda to adapt to the feature.
K = 500;
rng(1);
for i = 1:10
    [data4train, Gal, Prb, labelGal, labelPrb, para] = ...
        dataSplit_CUHK01(feature, 'single');
    para.labelTrain = []; % no need for label
    tic
    dist = CAMEL(data4train, K, lambda, Gal, Prb, para);
    fprintf('Running time of CAMEL: %f \n', toc);
    [~, rankingTable] = sort(dist);
    CMC_data4testAsym(:, i) = evalCMC(rankingTable, labelGal, labelPrb);
end
CMC_data4testAsym = mean(CMC_data4testAsym, 2);

rank1 = CMC_data4testAsym(1);
fprintf('rank1 = %f \r\n', rank1*100);