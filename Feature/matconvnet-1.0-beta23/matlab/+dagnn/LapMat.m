classdef LapMat < dagnn.Layer   
    properties
        K = 0
    end
    methods
        function outputs = forward(obj, inputs, params)
            % inputs: 1. feature,1*1*d*n; 2. idxView, 1*n
            feature = inputs{1};
            feature = squeeze(feature); % d*n
            idxViews = inputs{2};
            isGPU = isa(feature, 'gpuArray');
            feature = gather(feature); % single
            W = 1 - pdist2(feature', feature', 'cosine');
            uniViews = unique(idxViews);
            for i = 1:numel(uniViews)
                currentView = find(idxViews == uniViews(i));
                W(currentView, currentView) = 0;
            end
            if obj.K > 0
                for i = 1:size(W, 2)
                    list = W(:, i);
                    sortedList = sort(list, 'descend');
                    K_ = min(obj.K, size(W, 2));
                    minimal = sortedList(K_);
                    list(list < minimal) = 0;
                    W(:, i) = list;
                    W(i, :) = list';
                end
            end
                
            W(W < 0) = 0;
            Dcol = sum(W, 2);
            D = diag(Dcol);
            L = D - W;
            if isGPU
                L = gpuArray(L);
            end
            outputs{1} = L;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutpus)
            derInputs = cell(1,2);
            derParams = {};
        end
        
        function obj = LapMat(varargin)
            obj.load(varargin);
            obj.K = obj.K;
        end
            
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            n = inputSizes{1}(4);
            outputSizes{1} = [n, n];
        end
    end
    
end

