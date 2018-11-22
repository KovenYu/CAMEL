function U = constructU( V, D, nBasis, M, nViews )
% s.t. U'MU = NE
%   => u'Mu = N

% find the smallest eigens
d = diag(D);
[~, idx] = sort(d);
U_unnormalized = V(:, idx(1:nBasis));

U = zeros(size(U_unnormalized));
n = size(U, 2);
for i = 1:n
    u_un = U_unnormalized(:, i);
    u = u_un*sqrt(nViews)/sqrt(u_un'*M*u_un);
    U(:,i) = u;
end
U = real(U);
end

