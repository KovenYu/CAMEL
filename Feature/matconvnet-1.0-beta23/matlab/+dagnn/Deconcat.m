classdef Deconcat < dagnn.ElementWise
  methods
    function outputs = forward(obj, inputs, params)
      center = size(inputs{1}, 4) -10;
      outputs{1} = inputs{1}(:, :, :, 1:center);
      outputs{2} = inputs{1}(:, :, :, center+1 : end);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
%       derInputs = vl_nnconcat(inputs, obj.dim, derOutputs{1}, 'inputSizes', obj.inputSizes) ;
      derInput = inputs{1};
      center = size(inputs{1}, 4) -10;
      derInput(:, :, :, 1:center) = derOutputs{1};
      derInput(:, :, :, center+1 : end) = derOutputs{2};
      derInputs{1} = derInput;
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      sz = inputSizes{1} ;
      sz4 = size(sz, 4);
      center = sz4/2;
      sz(4) = center;
      outputSizes{1} = sz ;
    end

%     function rfs = getReceptiveFields(obj)
%       numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
%       if obj.dim == 3 || obj.dim == 4
%         rfs = getReceptiveFields@dagnn.ElementWise(obj) ;
%         rfs = repmat(rfs, numInputs, 1) ;
%       else
%         for i = 1:numInputs
%           rfs(i,1).size = [NaN NaN] ;
%           rfs(i,1).stride = [NaN NaN] ;
%           rfs(i,1).offset = [NaN NaN] ;
%         end
%       end
%     end

    function load(obj, varargin)
      s = dagnn.Layer.argsToStruct(varargin{:}) ;
      % backward file compatibility
      if isfield(s, 'numInputs'), s = rmfield(s, 'numInputs') ; end
      load@dagnn.Layer(obj, s) ;
    end

    function obj = Deconcat(varargin)
      obj.load(varargin{:}) ;
    end
  end
end
