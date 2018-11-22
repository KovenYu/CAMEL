function U = CAMEL_supervised( data, lambda, para )
% INPUT:
%
% data, the training data, arranged as a d-by-n matrix,
% where d is the dimension and n is the number of samples.
% 
% lambda, the cross-view regularizer.
%
% para, the parameters for the dataset, should contain at least:
% para.idxViewTrain, an array containing the index of views of the training
% samples.
% para.numViews, the number of cameara views. (typically 2)
% para.labelTrain, the labels of training images.
%
% OUTPUT:
% U, the transformation matrices, arranged as a (numViews*d)-by-d matrix.
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
%                                                     July. 2017.

[d, n] = size(data);
idxView = para.idxViewTrain;
nViews = para.numViews;
alpha = 1;
nBasis = d;
dataAsym = AsymShift(data, para, 'train');

IDX = para.labelTrain;
uni = unique(IDX);
for i = 1:numel(uni)
    IDX(IDX == uni(i)) = i;
end

L = IDX2L(IDX);

M = constructM(data, alpha, nViews, idxView);

I = constructI(d, nViews);
temp = (dataAsym*dataAsym' - dataAsym*(L*L')*dataAsym') / n;
MATRIX = M \ (lambda*I + temp);
[V,D] = eig(MATRIX);
U = constructU(V, D, nBasis, M, nViews);
