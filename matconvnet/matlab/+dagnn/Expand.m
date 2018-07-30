classdef Expand < dagnn.ElementWise
    properties 
        rate
    end

    methods
        function outputs = forward(obj, inputs, params)
            if obj.rate ~= 2
                error('rate other than 2 is not supported yet');
            end
            in = inputs{1};
            sz = size(in);
            osz = sz;
            osz(1:2) = osz(1:2) * 2;
            osz(3) = osz(3) / 4; 
            out = zeros(osz,'like',in);
            out(1:2:end,1:2:end,:,:)=in(:,:,            +1  : (sz(3)/4),:);
            out(1:2:end,2:2:end,:,:)=in(:,:,((sz(3)/4)  +1) : (sz(3)/2),:);
            out(2:2:end,1:2:end,:,:)=in(:,:,((sz(3)/2)  +1) : (sz(3)*3/4),:);
            out(2:2:end,2:2:end,:,:)=in(:,:,((sz(3)*3/4)+1) :  sz(3),:);
            outputs{1} = out;
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            in = inputs{1};
            sz = size(in);
            dI = zeros(sz, 'like', in); 
            dO = derOutputs{1};
            dI(:,:,            +1 :(sz(3)/4),:)   = dO(1:2:end,1:2:end,:,:);
            dI(:,:,((sz(3)/4)  +1):(sz(3)/2),:)   = dO(1:2:end,2:2:end,:,:);
            dI(:,:,((sz(3)/2)  +1):(sz(3)*3/4),:) = dO(2:2:end,1:2:end,:,:);
            dI(:,:,((sz(3)*3/4)+1): sz(3),:)      = dO(2:2:end,2:2:end,:,:);
            derInputs{1} = dI;
            derParams{1} = [];
        end

        function outputSizes = getOutputSizes(obj, inputSizes)
            if obj.rate ~= 2
                error('rate other than 2 is not supported yet');
            end
            isz = inputSizes{1};
            outputSizes{1} = [isz(1:2)*2,isz(3)/4,isz(4)];
        end

        function rfs = getReceptiveFields(obj)
            if obj.rate ~= 2
                error('rate other than 2 is not supported yet');
            end
            r = obj.rate;
            rfs.size = [1/r 1/r] ;
            rfs.stride = [1/r 1/r] ;
            rfs.offset = [1 1] ; % ???
        end

        function obj = Expand(varargin)
            obj.load(varargin) ;
        end
    end
end
