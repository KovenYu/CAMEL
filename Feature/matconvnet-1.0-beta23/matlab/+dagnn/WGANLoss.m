classdef WGANLoss < dagnn.ElementWise
  properties
    opts = {}
  end
  
  properties (Transient)
      average = 0
      numAveraged = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      input = inputs{1};
      input = squeeze(input); % now 1-dim
      input = -log(input);
      center = numel(input) / 2;
      l = sum(input(1 : center)) - sum(input(center+1 : end));
      outputs{1} = l;
      n = obj.numAveraged ;
      m = n + size(inputs{1},4)/2 ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derParams = [];
      input = inputs{1};
      derInput = -1./input;
      center = numel(derInput) / 2;
      derInput(center+1 : end) = - derInput(center+1 : end);
      derInputs{1} = derInput * derOutputs{1};
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

    function obj = WGANLoss(varargin)
      obj.load(varargin) ;
    end
  end
end
