classdef ResLoss < dagnn.ElementWise
  properties (Transient)
      average = 0
      numAveraged = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      X = inputs{1};
      X = squeeze(X); % d*n
      Y = inputs{2};
      Y = squeeze(Y);
      n_ = size(X, 2);
      outputs{1} = norm(X-Y, 'fro')^2 / n_;
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = zeros(size(inputs{1}), 'like', inputs{1}) ;
      derParams = {} ;
      
      X = inputs{1};
      X = squeeze(X); % d*n
      Y = inputs{2};
      derInput = Y; % 1*1*d*n
      Y = squeeze(Y);
      n_ = size(X, 2);
      derInput(1, 1, :, :) = 2*(Y-X) /n_;
      derInputs{2} = derInput * derOutputs{1};
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 1] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = ResLoss(varargin)
      obj.load(varargin) ;
    end
  end
end
