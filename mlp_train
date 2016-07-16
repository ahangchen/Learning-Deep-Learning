function [ outputs, weight1, bias1, weight2, bias2, MSE ] = mlp_train( inputs, targets)
%MLP 三层感知机，使用反向传播算法训练
% 初始化网络参数
learning_rate = 0.1;
hiddenlayers = 6;
max_iter = 1000;
epsilon = 0.01;

% 初始化权重和偏置
weight1 = 0.5^0.5*randn(size(inputs,1),hiddenlayers);
bias1 = zeros(hiddenlayers, 1);
weight2 = 0.5^0.5*randn(hiddenlayers, size(targets,1));
bias2 = zeros(size(targets,1),1);
MSE = zeros(max_iter,1);
for i=1:max_iter
    % 每次迭代随机打乱数据
    orders = randperm(size(inputs,2));
    inputs = inputs(:,orders);
    targets = targets(orders);
    
    error = 0;
    for Q = 1:size(inputs,2)
        %前向传播
        out_linear = weight1' * inputs(:, Q) + bias1;
        out_nonlinear = 1 ./ (1 + exp(-out_linear));
        outputs = weight2' * out_nonlinear +bias2;
        loss = 1/2 * (outputs - targets(Q))^2;
        error = error + loss;
        
        %反向传播
        top_diff = outputs-targets(Q);
        
        %更新权值 weight2
        weight2 = weight2 - learning_rate.*(out_nonlinear*top_diff');
        bias2 = bias2 - 2*learning_rate.*top_diff;
        
        %反向传播导数 对上一层的输入求导
        top_diff = weight2 * top_diff.*(1./(exp(out_linear)+exp(0-out_linear)+2));
        
        %更新权值 weight1
        weight1 = weight1 - learning_rate.*(inputs(:, Q)*top_diff');
        bias1= bias1 - 2*learning_rate.*top_diff;
    end
    MSE(i) = error/size(inputs,2);
    % 输出平均方差
    fprintf('Iteration %d: MSE=%d \n',i,error/size(inputs,2));
    
    % 检查是否达到精度
    if error < epsilon
        fprintf('Optimization done.\n');
        break;
    end
    
end

end
