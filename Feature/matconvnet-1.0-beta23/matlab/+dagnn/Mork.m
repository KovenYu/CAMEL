classdef Mork < dagnn.Filter
  properties
    size = [0 0 0 0]
    opts = {'cuDNN'}
    initMethod = 'gaussian'
    nViews = 2
    lambda = 0
    gamma = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
        % inputs: 1. feature(e.g. fc1) 2. idxViews of the batch 3. M
        % params: n. the n-th view's U
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
              input, params{idxView}, [], ...
              'pad', obj.pad, ...
              'stride', obj.stride, ...
              'dilate', obj.dilate, ...
              obj.opts{:});
      end
      outputs{1} = out;
      
%       [~, ~, Nd, T] = size(params{1});
%         U1 = reshape(params{1}, [Nd, T]);
%         U2 = reshape(params{2}, [Nd, T]);
%         U = [U1; U2];
%         M = inputs{3};
%         cLoss = norm(U'*M*U-2*eye(T) ,'fro')^2;
%         uLoss = norm(U1-U2, 'fro')^2;
%         fprintf('\n gamma*cLoss = %f, lambda*uLoss = %f\n', obj.gamma*cLoss, obj.lambda*uLoss);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        % inputs: 1. feature(e.g. fc1) 2. idxViews of the batch 3. M
        % params: n. the n-th view's U.
        derInput = inputs{1};
        derParams = cell(1, obj.nViews);
        for idxView = 1:obj.nViews
            idxViews = inputs{2};
            batchInput = inputs{1};
            idxImgCurrentView = (idxViews == idxView);
            if isempty(idxImgCurrentView), continue, end
            input = batchInput(:, :, :, idxImgCurrentView);
            derOutput = derOutputs{1};
            derOutput = derOutput(:, :, :, idxImgCurrentView);
            [derInput(:, :, :, idxImgCurrentView), derParams{idxView}] = ...
                vl_nnconv( ...
                input, params{idxView}, [], derOutput, ...
                'pad', obj.pad, ...
                'stride', obj.stride, ...
                'dilate', obj.dilate, ...
                obj.opts{:});
        end
        
        derInputs{1} = derInput;
        derInputs{2} = [];
        derInputs{3} = [];
        
%         % note that this is for 2 Views version.
%         [~, ~, Nd, T] = size(params{1});
%         U1 = reshape(params{1}, [Nd, T]);
%         U2 = reshape(params{2}, [Nd, T]);
%         U = [U1; U2];
%         M = inputs{3};
%         der = obj.gamma*(4*M*U*(U'*M*U-2*eye(T)));
%         derParam = der(1:Nd, :);
%         derParam = reshape(derParam, [1, 1, Nd, T]);
%         derParams{1} = derParams{1} + derParam;
%         derParam = der(Nd+1:end, :);
%         derParam = reshape(derParam, [1, 1, Nd, T]);
%         derParams{2} = derParams{2} + derParam;
%         % regularization term
%         for i = 1:obj.nViews % obj.nViews = 2
%             j = setdiff(1:obj.nViews, i);
%             % der for U
%             der = obj.lambda*(params{i} - params{j});
%             derParams{i} = derParams{i} + der;
%         end
          %% this is for n Views version
          [~, ~, Nd, T] = size(params{1});
          U = [];
          for i = 1:obj.nViews
              Ui = reshape(params{i}, [Nd, T]);
              U = [U; Ui];
          end
          M = inputs{3};
          der = obj.gamma*(4*M*U*(U'*M*U-2*eye(T)));
          for i = 1:obj.nViews
              derParam = der(1+(i-1)*Nd: i*Nd, :);
              derParam = reshape(derParam, [1, 1, Nd, T]);
              derParams{i} = derParams{i} + derParam;
          end
          % regularization term
          dim = Nd;
          E = eye(dim);
          minusBigE = repmat(-E, obj.nViews, obj.nViews);
          I = minusBigE + obj.nViews*eye(obj.nViews*Nd);
          der = obj.lambda*2*I*U;
          for i = 1:obj.nViews
              derParam = der(1+(i-1)*Nd: i*Nd, :);
              derParam = reshape(derParam, [1, 1, Nd, T]);
              derParams{i} = derParams{i} + derParam;
          end
              
    end
    
    function setParams(obj, newParams)
             % newParams : 1* nViews cell.
       assert(numel(newParams) == obj.nViews);
       theLayer = obj.net.layers(obj.layerIndex);
       paramsNames = theLayer.params;
       assert(numel(paramsNames) == numel(newParams));
       for i = 1:obj.nViews
           idxPar = obj.net.getParamIndex(paramsNames{i});
           obj.net.params(idxPar).value = newParams{i};
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

    function obj = Mork(varargin)
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
