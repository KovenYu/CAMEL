function U = CAMEL( data, k, lambda, para )
% INPUT:
%
% data, the training data, arranged as a d-by-n matrix,
% where d is the dimension and n is the number of samples.
%
% k, the number of centroids.
% 
% lambda, the cross-view regularizer.
%
% para, the parameters for the dataset, should contain at least:
% para.idxViewTrain, an array containing the index of views of the training
% samples.
% para.numViews, the number of cameara views. (typically 2)
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
maxIter = 100;
options = statset('MaxIter', 100);
dataAsym = AsymShift(data, para, 'train');

IDX = kmeans(data', k, 'options', options);

L = IDX2L(IDX);

M = constructM(data, alpha, nViews, idxView);

I = constructI(d, nViews);
temp = (dataAsym*dataAsym' - dataAsym*(L*L')*dataAsym') / n;
MATRIX = M \ (lambda*I + temp);
[V,D] = eig(MATRIX);
U = constructU(V, D, nBasis, M, nViews);

fU = lambda*trace(U'*I*U);
fclustering = (trace((U*U')*(dataAsym*dataAsym')) - trace(L'*dataAsym'*(U*U')*dataAsym*L)) / n;
fobj = fU + fclustering;

%% iterative optimization
fobjPrevious = fobj;
for iter = 1:maxIter
    XprojT = (U'*dataAsym)';
    
    IDX = kmeans(XprojT, k, 'options', options);
    
    L = IDX2L(IDX);
    if any(any(isnan(L))) % a matlab bug.
        IDX = kmeans(XprojT, k , 'options', options);
        L = DIX2L(IDX);
    end
        
    MATRIX = M \ (lambda*I +(dataAsym*dataAsym' - dataAsym*(L*L')*dataAsym') / n);
    
    [V,D] = eig(MATRIX);
    U = constructU(V, D, nBasis, M, nViews);
    
    fU = lambda*trace(U'*I*U);
    fclustering = (trace(dataAsym'*(U*U')*dataAsym) - trace(L'*dataAsym'*(U*U')*dataAsym*L)) / n;
    fobj = fU + fclustering;
    
    if (fobj - fobjPrevious) > 0
        break
    end
    fobjPrevious = fobj;
    if iter == maxIter
        warning('The default maximum # of iterations has been reached.');
    end

end

