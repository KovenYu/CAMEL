% If this code is helpful in your research, please kindly cite our work:
%
% Hong-Xing Yu, Ancong Wu, Wei-Shi Zheng,
% Cross-view Asymmetric Metric Learning for 
% Unsupervised Person Re-identification,
% Proceedings of the IEEE International Conference on Computer Vision,
% 2017.
%
%                                                    Hong-Xing Yu
%                                                     July. 2017.

clear
clc
warning off all
load('data/Market_JSTL64');
addpath('utils')
lambda = 1e-2;
% if you use different feature other than JSTL-64, you may want to
% set a larger lambda e.g. 1e-1 to adapt to the feature.
rng(1);
tic
fprintf('CAMEL_supervised running. This may take several minutes in Market-1501 ..\n')
U = CAMEL_supervised(data4train, lambda, para);
fprintf('Running time of CAMEL_supervised: %f seconds \n', toc);

fprintf('Evaluating .. \r\n')
data4testAsym = AsymShift(data4test, para, 'test');
galAsym = U'*data4testAsym;
queryAsym = U'*AsymShift(data4query, para, 'query');

dist = pdist2(galAsym', queryAsym','cosine');
% cosine distance is equivalent to L2 distance when the transformed
% feature is L2 normalized.
[CMC, MAP] = evalCMCnMAP(dist, para);

rank1 = CMC(1);
rank5 = CMC(5);
rank10 = CMC(10);
fprintf('Rank1 = %.2f, Rank5 = %.2f, Rank10 = %.2f, MAP = %.2f \r\n', ...
    rank1*100, rank5*100, rank10*100, MAP*100);