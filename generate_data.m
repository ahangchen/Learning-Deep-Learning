function [ X, Y, inputs, targets ] = generate_data( ~ )
%����1000000������[-1��1]�е�������������ɵ�Ԫ��
%   �˴���ʾ��ϸ˵��
% inputs = rand(2, 100000) * 2 - 1;
[X, Y] = meshgrid(-1:0.02:1, -1:0.02:1);
% [m n] = size(X);
% num = m*n;
% X = reshape(X,[1 num]); Y = reshape(Y, [1 num]);
inputs = [reshape(X,[1,size(X,1)*size(X,2)]);
    reshape(Y,[1,size(Y,1)*size(Y,2)])];
targets1 = abs( inputs(1, :) ) < 0.5 ;
targets2 = abs( inputs(2, :) ) < 0.5 ;
targets = targets1 + targets2;
save( 'mlp_count_data.mat', 'inputs', 'targets' );
end

