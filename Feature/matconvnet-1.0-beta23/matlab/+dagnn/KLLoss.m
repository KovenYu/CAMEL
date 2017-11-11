classdef KLLoss < dagnn.ElementWise
  properties
    opts = {}
  end
  
  properties (Transient)
      average = 0
      numAveraged = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = Kr_KLLoss(inputs{1}, inputs{2}, params{1}, []);
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      [derInputs{1}, derParams{1}] = Kr_KLLoss(inputs{1}, inputs{2}, params{1}, derOutputs);
      derInputs{2} = [];
    end

    function setParams(obj, MU)
       theLayer = obj.net.layers(obj.layerIndex);
       paramsNames = theLayer.params;
       idxPar = obj.net.getParamIndex(paramsNames{1});
       obj.net.params(idxPar).value = MU;
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

    function obj = KLLoss(varargin)
      obj.load(varargin) ;
    end
  end
end
