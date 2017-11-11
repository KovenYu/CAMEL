classdef GANLoss < dagnn.ElementWise
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
      input = input(:, :, 1, :);
      input = squeeze(input); % now 1-dim
      center = numel(input) / 2;
      inputx = log(input(1:center));
      inputz = log(1-input(center+1 : end));
      l = sum(inputx) + sum(inputz);
      outputs{1} = l;
      n = obj.numAveraged ;
      m = n + size(inputs{1},4)/2 ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      input = inputs{1};
      input = input(:, :, 1, :);
      input = squeeze(input); % now 1-dim
      center = numel(input) / 2;
      dinputx = 1./(input(1:center));
      dinputz = 1./(input(center+1 : end)-1);
      derInput = inputs{1};
      derInput(1, 1, 1, 1:center) = dinputx;
      derInput(1, 1, 1, center+1 : end) = dinputz;
      derInput(1, 1, 2, :) = zeros(1,1,1,center*2, 'like', derInput);
      derInputs{1} = derInput * derOutputs{1};
      derParams = [];
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

    function obj = GANLoss(varargin)
      obj.load(varargin) ;
    end
  end
end
