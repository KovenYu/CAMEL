function L = IDX2L( IDX )

% input   IDX, 1*n, every element denotes which centroid this element
% belongs to
% output  L, n*k

k = unique(IDX);
k = k(end);
n = length(IDX);
L = zeros(n,k);
for i = 1:k
    temp = IDX == i;
    L(:,i) = temp/ sqrt(sum(temp));
end

