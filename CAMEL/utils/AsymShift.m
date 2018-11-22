function shiftedData = AsymShift( originalData, para, phase )
% input :   originalData,  d by n
% output:   shiftedData,  para.numViews*d by n
% shift training data to fit the combined projection matrix U

d = para.dimFea;
if strcmp(phase, 'train')
    n = para.numTotalImgTrain;
    idxView = para.idxViewTrain;
elseif strcmp(phase, 'test')
    n = para.numTotalImgTest;
    idxView = para.idxViewTest;
elseif strcmp(phase, 'query')
    n = para.numTotalImgQuery;
    idxView = para.idxViewQuery;
else
    error('phase must be either train or test (string) \n');
end
    
numViews = para.numViews;
shiftedData = zeros(numViews*d, n);
for i = 1:numViews;
    idx = (idxView == i);
    l = sum(idx); % number of the imgs under the i-th views
    if i == 1
        data = [originalData(:,idx); zeros( (numViews-1)*d, l)];
    elseif i == numViews
        data = [zeros( (numViews-1)*d, l); originalData(:,idx)];
    else
        data = [zeros( (i-1)*d, l); originalData(:,idx); zeros( (numViews-i)*d, l)];
    end
    shiftedData(:,idx) = data;
end
end