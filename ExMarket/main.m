clear
clc
load('ExMarket_JSTL64.mat');
dist = pdist2(data4test', data4query');
% Euclidean distance which is suggested by the original JSTL model as a baseline
[CMC, Map] = evalCMCnMAP(dist, para);
rank1 = CMC(1);
fprintf('rank1 = %f, MAP = %f \n', rank1, Map);