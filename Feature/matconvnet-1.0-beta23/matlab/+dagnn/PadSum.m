classdef PadSum < dagnn.Sum
  %PADSUM DagNN sum layer with automatic feature padding
  %   The PADSUM layer takes the sum of all its inputs and store the result
  %   as its only output. Inputs with insufficient depth will be zero padded

  properties (Transient)
  end

  methods
    function outputs = forward(obj, inputs, params)
      nCh = size(inputs{1},3);
      obj.numInputs = numel(inputs) ;
      for k = 1:obj.numInputs, 
        if size(inputs{k},3)>nCh, nCh = size(inputs{k},3); end
      end
      outputs{1} = zeros(size(inputs{1},1),size(inputs{1},2),nCh,size(inputs{1},4), ...
        'like', inputs{1});
      for k = 1:obj.numInputs
        nCh = size(inputs{k},3);
        outputs{1}(:,:,1:nCh,:) = outputs{1}(:,:,1:nCh,:) + inputs{k} ;
      end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      for k = 1:obj.numInputs
        derInputs{k} = derOutputs{1}(:,:,1:size(inputs{k},3),:) ;
      end
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = inputSizes{1} ;
      for k = 2:numel(inputSizes)
        if inputSizes{k}(3)>=outputSizes{1}(3), 
          outputSizes{1}(3) = inputSizes{k}(3);
        end
        if all(~isnan(inputSizes{k})) && all(~isnan(outputSizes{1}))
          if ~isequal(inputSizes{k}([1:2,4:end]), outputSizes{1}([1:2,4:end])),
            warning('PadSum layer: the 1&2&4 dimensions of the input variables are not the same.') ;
          end
        end
      end
    end

    function obj = PadSum(varargin)
      obj.load(varargin) ;
    end
  end
end