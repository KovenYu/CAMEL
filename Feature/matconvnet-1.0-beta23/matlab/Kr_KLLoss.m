function [ output1, output2 ] = Kr_KLLoss( input, P, MU, derOutput )

Z = squeeze(input);
iMax = size(Z, 2);
jMax = size(MU, 2);
d = size(Z, 1);
sigma = 200;

Z = gather(Z);
MU = gather(MU);
P = gather(P);

Q = zeros(iMax, jMax, 'like', Z);
% eval Q
for i = 1:iMax
    z = Z(:, i);
    denominator = 0;
    for j = 1:jMax
        mu = MU(:, j);
%         denominator = denominator + (1 + norm(z-mu)^2)^(-1);
        denominator = denominator + exp(-norm(z-mu)^2/sigma);
    end
    for j = 1:jMax
        mu = MU(:, j);
        Q(i, j) = exp(-norm(z-mu)^2/sigma) / denominator;
    end
end

if isempty(derOutput) % forward
    LOG = log(P./Q);
    KL = P.*LOG;
    loss = sum(sum(KL));
    output1 = gpuArray(loss);
    output2 = {};

else % backward
    wholeGrad = zeros(iMax, jMax, d);
    for i = 1:iMax
        z = Z(:, i);
        for j = 1:jMax
            mu = MU(:, j);
            p = P(i, j);
            q = Q(i, j);
%             wholeGrad(i, j, :) = (1+norm(z-mu)^2)^(-1)*(p-q)*(z-mu);
            wholeGrad(i, j, :) = (p-q)*(z-mu);
        end
    end
    gradZ = squeeze(sum(wholeGrad, 2));
    gradZ = gradZ';
    gradZ = reshape(gradZ, [1, 1, size(gradZ)]);
    gradMU = squeeze(sum(wholeGrad, 1));
    gradMU = -gradMU';
    output1 = gpuArray(single(gradZ) * derOutput{1});
    output2 = gpuArray(single(gradMU) * derOutput{1});
end

end

