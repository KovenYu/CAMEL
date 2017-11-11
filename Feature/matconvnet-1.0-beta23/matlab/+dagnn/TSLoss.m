classdef TSLoss < dagnn.ElementWise
  properties
    opts = {}
    nTrans
  end
  
  properties (Transient)
      average = 0
      numAveraged = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      input = squeeze(inputs{1}); % d*n
      l = 0;
      fold = size(input, 2)/(obj.nTrans+1);
      for i = 1:obj.nTrans
          if mod(fold, 1) ~= 0, break, end
          idx1 = (i-1)*fold+1;
          idx2 = (i-1)*fold+fold;
          for j = i+1:obj.nTrans+1
              idx3 = (j-1)*fold+1;
              idx4 = (j-1)*fold+fold;
              matrix = input(:, idx1:idx2) - input(:, idx3:idx4);
              l = l+matrix(:)'*matrix(:);
          end
      end
      outputs{1} = l;
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInput = zeros(size(inputs{1}), 'like', inputs{1});
      input = squeeze(inputs{1}); % d*n
      fold = size(input, 2)/(obj.nTrans+1);
      for i = 1:obj.nTrans           
          idx1 = (i-1)*fold+1;
          idx2 = (i-1)*fold+fold;
          for j = i+1:obj.nTrans+1
              idx3 = (j-1)*fold+1;
              idx4 = (j-1)*fold+fold;
              matrix = input(:, idx1:idx2) - input(:, idx3:idx4);
              temp = zeros(size(derInput), 'like', derInput);
              temp(1, 1, :, idx1:idx2) = matrix;
              derInput = derInput + temp;
          end
      end
      derInputs{1} = derInput*derOutputs{1};
      derParams = {};
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

    function obj = TSLoss(varargin)
      obj.load(varargin) ;
    end
  end
end
