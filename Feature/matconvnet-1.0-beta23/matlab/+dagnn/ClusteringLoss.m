classdef ClusteringLoss < dagnn.ElementWise
  properties
    opts = {}
  end
  
  properties (Transient)
      average = 0
      numAveraged = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      % inputs: 1.fork, 1*1*k*n; 2.centroids, k*K; 3.IDX, 1*n
      fork = inputs{1};
      centroids = inputs{2};
      IDX = inputs{3};
      fork = squeeze(fork); % k*n
%       assert(numel(IDX) == size(fork, 2));
      l = zeros(1, 1, 'like', fork);
      for i = 1:size(fork, 2)
          idx = IDX(i);
          l = l + (norm(fork(:, i) - centroids(:, idx)))^2;
      end
      outputs{1} = l;
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{2} = [] ;
      derInputs{3} = [] ;
      derParams = {} ;
      
      fork = inputs{1};
      centroids = inputs{2};
      IDX = inputs{3};
      derInput = fork; % 1*1*k*n
      fork = squeeze(fork); % k*n
      for i = 1:size(fork, 2)
          idx = IDX(i);
          derInput(:, :, :, i) = 2*(fork(:, i) - centroids(:, idx)) * derOutputs{1};
      end
      derInput = derInput / numel(IDX);
      derInputs{1} = derInput;
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = ClusteringLoss(varargin)
      obj.load(varargin) ;
    end
  end
end
