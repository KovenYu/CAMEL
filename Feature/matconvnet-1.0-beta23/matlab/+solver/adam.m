function [w, state] = adam(w, state, grad, opts, lr)


if nargin == 0 % Return the default solver options
  w = struct('epsilon', 1e-8, 'beta1', 0.9, 'beta2', 0.999);
  return;
end

if isempty(state)
  state{1} = zeros(size(grad), 'like', grad); % m
  state{2} = zeros(size(grad), 'like', grad); % v
  state{3} = 1; % t
  state = cellfun(@gpuArray, state, 'uniformoutput', false) ;
end

m = opts.beta1*state{1} + (1 - opts.beta1)*grad;
v = opts.beta2*state{2} + (1 - opts.beta2)*(grad.^2);
if state{3} < 50
    mb = m/(1 - opts.beta1^state{3});
else
    mb = m; 
end
vb = v/(1 - opts.beta2^state{3});

% note that lr is required to be set relatively small, e.g. lr = 0.001.
w = w - lr * mb ./ (sqrt(vb) + opts.epsilon) ;


state{1} = m;
state{2} = v;
state{3} = state{3} + 1;
