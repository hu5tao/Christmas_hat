classdef HuberLoss < dagnn.ElementWise
    properties
        loss = 'huber'
        opts = {}
    end
    properties (Transient)
        average = 0
        numAveraged = 0
    end

    methods
        function outputs = forward(obj, inputs, params)
            %pred_reg = inputs{1}; 
            %label_reg = inputs{2}; 
            %label_cls = inputs{3}; 
            
            %pos = repmat(label_cls>0,[1,1,4,1]);
            %a = abs(pred_reg - label_reg); 
            %b = (a < 1); 
            %t = (b.*(a.^2))*0.5 + (~b).*(a-0.5);
            %outputs{1} = sum(t(pos));
                
            % compute the loss after we finish hard example mining
            outputs{1} = 0;
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derOutput = derOutputs{1}; 
            
            pred_reg = inputs{1}; 
            label_reg = inputs{2}; 
            label_cls = inputs{3}; 
            
            pos = repmat(label_cls>0,[1,1,4,1]);
            x = pred_reg - label_reg;
            der = (1<=x) + (x<=-1)*(-1) + (-1<x&x<1).*x;
            derInputs{1} = derOutput*(der).*(pos);
            derInputs{2} = [] ;
            derInputs{3} = [] ; 
            derParams = {} ;

            % now we have an updated label_cls from hard example mining
            pos = repmat(label_cls>0,[1,1,4,1]);
            a = abs(pred_reg - label_reg); 
            b = (a < 1); 
            t = (b.*(a.^2))*0.5 + (~b).*(a-0.5);
            loss = sum(t(pos));

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
            rfs(3,1) = rfs(1,1) ;
        end

        function obj = HuberLoss(varargin)
            obj.load(varargin) ;
        end
    end
end
