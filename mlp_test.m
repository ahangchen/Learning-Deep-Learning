function [ outputs ] = mlp_test( inputs, weight1, bias1, weight2, bias2 )
%MLP_TEST 此处显示有关此函数的摘要
%   此处显示详细说明
outputs = weight2'*(1./(1+exp(0-(weight1'*inputs + repmat(bias1,[1,size(inputs,2)]))))) + repmat(bias2,[1,size(inputs,2)]);

end

