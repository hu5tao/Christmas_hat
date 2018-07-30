classdef DetLoss < dagnn.ElementWise
  properties
      loss = 'softmaxlog'
    opts = {}
  end

  properties (Transient)
    average = 0
    numAveraged = 0
    loss_map = []
  end

  methods
    function outputs = forward(obj, inputs, params)
        [outputs{1}, obj.loss_map] = vl_nnloss_detail(inputs{1}, inputs{2}, [], ...
                                                      'loss', obj.loss, obj.opts{:}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnloss_detail(inputs{1}, inputs{2}, derOutputs{1}, 'loss', obj.loss, obj.opts{:}) ;
      derInputs{2} = [] ;
      derParams = {} ;

      %% keep track of the loss based on selected hard examples
      valid = single(inputs{2}(:)~=0);
      loss = valid(:)' * obj.loss_map(:);
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(loss)) / m ;
      obj.numAveraged = m ;
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

    function obj = DetLoss(varargin)
      obj.load(varargin) ;
    end
  end
end
