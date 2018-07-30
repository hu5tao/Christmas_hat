classdef RegLoss < dagnn.ElementWise
    properties
        loss = 'huber'
        opts = {}
    end
    properties (Transient)
        average = 0
        numAveraged = 0
        
        instanceWeights = []
        sampleSize = 0
    end

    methods
        function outputs = forward(obj, inputs, params)
            X = inputs{1}; 
            cr = inputs{2}; 
            cc = inputs{3};
            w = logical(cc==1);
            w = repmat(w, [1,1,size(cr,3)/size(cc,3),1]);
            obj.instanceWeights = w;
            obj.sampleSize = sum(w(:));
            
            a = abs(X - cr); 
            b = (a < 1); 
            t = (b.*(a.^2))*0.5 + (~b).*(a-0.5);
            %outputs{1} = sum(t(w)) / obj.sampleSize;
            outputs{1} = sum(t(w));
            
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            X = inputs{1}; 
            cr = inputs{2}; 
            cc = inputs{3}; 
            
            w = obj.instanceWeights;
            x = X - cr;
            Y = (1<=x) + (x<=-1)*(-1) + (-1<x&x<1).*x;
            %derInputs{1} = (Y).*(w) / obj.sampleSize;
            derInputs{1} = (Y).*(w);
            
            derInputs{2} = [] ;
            derInputs{3} = [] ; 
            derParams = {} ;
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
            rfs(3,1) = rfs(1,1) ;
        end

        function obj = RegLoss(varargin)
            obj.load(varargin) ;
        end
    end
end
