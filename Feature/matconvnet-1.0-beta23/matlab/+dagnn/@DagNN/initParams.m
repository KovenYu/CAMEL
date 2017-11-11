function initParams(obj, targetLayers)
% INITPARAM  Initialize the paramers of the DagNN
%   OBJ.INITPARAM() uses the INIT() method of each layer to initialize
%   the corresponding parameters (usually randomly).

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if nargin < 2 || isempty(targetLayers)
    for l = 1:numel(obj.layers)
        p = obj.getParamIndex(obj.layers(l).params) ;
        params = obj.layers(l).block.initParams() ;
        switch obj.device
            case 'cpu'
                params = cellfun(@gather, params, 'UniformOutput', false) ;
            case 'gpu'
                params = cellfun(@gpuArray, params, 'UniformOutput', false) ;
        end
        [obj.params(p).value] = deal(params{:}) ;
    end
else
    layerIdxs = obj.getLayerIndex(targetLayers);
    for i = 1:numel(layerIdxs)
        l = layerIdxs(i);
        p = obj.getParamIndex(obj.layers(l).params) ;
        params = obj.layers(l).block.initParams() ;
        switch obj.device
            case 'cpu'
                params = cellfun(@gather, params, 'UniformOutput', false) ;
            case 'gpu'
                params = cellfun(@gpuArray, params, 'UniformOutput', false) ;
        end
        [obj.params(p).value] = deal(params{:}) ;
    end
end