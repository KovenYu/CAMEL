classdef Fork < dagnn.Filter
  properties
    size = [0 0 0 0]
    opts = {'cuDNN'}
    initMethod = 'gaussian'
    nViews = 2
    lambda = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
        % inputs: 1. feature(e.g. fc1) 2. idxViews of the batch
        % params: n. the n-th view's U, n+obj.nViews. b
      if numel(inputs) ~= 2
          error('inputs must contain the data and the idxViews of the data.\n')
      end
      inputSize = {size(inputs{1})};
      outputSize = obj.getOutputSizes(inputSize);
      out = zeros(outputSize{1}, 'like', inputs{1});
      for idxView = 1:obj.nViews
          idxViews = inputs{2};
          batchInput = inputs{1};
          idxImgCurrentView = (idxViews == idxView);
          if isempty(idxImgCurrentView), continue, end
          input = batchInput(:, :, :, idxImgCurrentView);
          out(:, :, :, idxImgCurrentView) = vl_nnconv(...
              input, params{idxView}, params{idxView + obj.nViews}, ...
              'pad', obj.pad, ...
              'stride', obj.stride, ...
              'dilate', obj.dilate, ...
              obj.opts{:});
      end
      outputs{1} = out;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        % inputs: 1. feature(e.g. fc1) 2. idxViews of the batch
        % params: n. the n-th view's U.
        derInput = inputs{1};
        derParams = cell(1, obj.nViews*2);
        for idxView = 1:obj.nViews
            idxViews = inputs{2};
            batchInput = inputs{1};
            idxImgCurrentView = (idxViews == idxView);
            if isempty(idxImgCurrentView), continue, end
            input = batchInput(:, :, :, idxImgCurrentView);
            derOutput = derOutputs{1};
            derOutput = derOutput(:, :, :, idxImgCurrentView);
            [derInput(:, :, :, idxImgCurrentView), derParams{idxView},...
                derParams{idxView + obj.nViews}] = ...
                vl_nnconv( ...
                input, params{idxView}, params{idxView + obj.nViews}, derOutput, ...
                'pad', obj.pad, ...
                'stride', obj.stride, ...
                'dilate', obj.dilate, ...
                obj.opts{:});
        end
        derInputs{1} = derInput;
        derInputs{2} = [];
        
        % note that this is for 2 Views version.
        % regularization term
        for i = 1:obj.nViews % obj.nViews = 2
            j = setdiff(1:obj.nViews, i);
            % der for U
            der = obj.lambda*(params{i} - params{j});
            derParams{i} = derParams{i} + der;
            % der for B
            der = obj.lambda*(params{i + obj.nViews} - params{j + obj.nViews});
            derParams{i + obj.nViews} = derParams{i + obj.nViews} + der;
        end
    end
    
    function setParams(obj, newParams)
             % newParams : 1* 2nViews cell.
       assert(numel(newParams) == 2*obj.nViews);
       theLayer = obj.net.layers(obj.layerIndex);
       paramsNames = theLayer.params;
       assert(numel(paramsNames) == numel(newParams));
       for i = 1:obj.nViews
           idxPar = obj.net.getParamIndex(paramsNames{i});
           obj.net.params(idxPar).value = newParams{i};
           idxPar = obj.net.getParamIndex(paramsNames{i + obj.nViews});
           obj.net.params(idxPar).value = newParams{i + obj.nViews};
       end
    end
    
    function params = initParams(obj)
      % Xavier improved
      sc = sqrt(2 / prod(obj.size(1:3))) ;
      %sc = sqrt(2 / prod(obj.size([1 2 4]))) ;
      switch obj.initMethod
          case 'gaussian'
              for i = 1:obj.nViews
                  params{i} = randn(obj.size,'single') * sc ;
              end
          case 'one'
              for i = 1:obj.nViews
                  params{i} = permute(eye(obj.size(3), 'single'), [3,4,1,2,]);
              end
      end
      for i = 1:obj.nViews
          params{i + obj.nViews} = zeros([obj.size(3), 1], 'single');
      end
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = obj.size(4) ;
    end

    function set.size(obj, ksize)
      % make sure that ksize has 4 dimensions
      ksize = [ksize(:)' 1 1 1 1] ;
      obj.size = ksize(1:4) ;
    end

    function obj = Fork(varargin)
      obj.load(varargin) ;
      % normalize field by implicitly calling setters defined in
      % dagnn.Filter and here
      obj.size = obj.size ;
      obj.stride = obj.stride ;
      obj.pad = obj.pad ;
      obj.nViews = obj.nViews;
      obj.lambda = obj.lambda;
    end
  end
end
