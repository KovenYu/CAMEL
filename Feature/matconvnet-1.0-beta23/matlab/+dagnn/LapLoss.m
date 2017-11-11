classdef LapLoss < dagnn.ElementWise
  
  properties (Transient)
      average = 0
      numAveraged = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      % inputs: 1.Y, 1*1*d*n; 2.L, n*n
      Y = inputs{1};
      L = inputs{2};
      Y = squeeze(Y); % d*n
      ny = numel( find(L < 0) );
      outputs{1} = trace(Y*L*Y') /ny;
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{2} = zeros(size(inputs{2}), 'single') ;
      derParams = {} ;
      
      Y = inputs{1};
      L = inputs{2};
      ny = numel( find(L < 0) );
      derInput = Y; % 1*1*d*n
      Y = squeeze(Y); % d*n
      derInput(1, 1, :, :) = 2*Y*L /ny;
      derInputs{1} = derInput * derOutputs{1};
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

    function obj = LapLoss(varargin)
      obj.load(varargin) ;
    end
  end
end
