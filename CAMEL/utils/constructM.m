function M = constructM( data, alpha, numViews, idxView )
% input :  data,  d by n
% output:  M = [M1 0  ... 0
%               0  M2 ... 0
%               0  0  ... Mv]
%        where Mi = cov of data in i-th view.  d by d

d = size(data, 1);
M = zeros(d*numViews);

for i = 1:numViews
    idx = (idxView == i);
    tempX = data(:, idx);
    ni = size(tempX, 2);
    M(1 + (i-1)*d : i*d, 1 + (i-1)*d : i*d) = tempX*tempX' / ni; % Mi
end

M = M + alpha*trace(M)/size(M,1)*eye(size(M));