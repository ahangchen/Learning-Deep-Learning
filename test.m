[ X, Y, inputs, targets ] = generate_data();
 [ outputs, weight1, bias1, weight2, bias2, MSE ] = mlp_train(inputs, targets);

% mlp_count;
% [X, Y] = meshgrid(-1:0.02:1, -1:0.02:1);
% [m, n] = size(X);
% num = m*n;
% X = reshape(X,[1 num]); Y = reshape(Y, [1 num]);
% test_data = [X;Y];
Z = mlp_test(inputs,  weight1, bias1, weight2, bias2);
% X = reshape(X,[m n]);
% Y = reshape(Y, [m n]);
Z = reshape(Z,size(X));
figure(1);
mesh(X,Y,Z);
