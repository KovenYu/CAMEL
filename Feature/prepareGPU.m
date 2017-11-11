function prepareGPU(gpus)
cold = true;
numGpus = numel(gpus) ;
if numGpus > 1
  % check parallel pool integrity as it could have timed out
  pool = gcp('nocreate') ;
  if ~isempty(pool) && pool.NumWorkers ~= numGpus
    delete(pool) ;
  end
  pool = gcp('nocreate') ;
  if isempty(pool)
    parpool('local', numGpus) ;
    cold = true ;
  end

end
if numGpus >= 1 && cold
  fprintf('%s: resetting GPU\n', mfilename)
  clearMex() ;
  if numGpus == 1
    gpuDevice(gpus)
  else
    spmd
      clearMex() ;
      gpuDevice(gpus(labindex))
    end
  end
end

% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
clear vl_tmove vl_imreadjpeg ;