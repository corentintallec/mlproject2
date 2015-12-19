function [cost, grad] = softmaxCost(beta, numClasses, inputSize, lambda, data, labels)

% beta - numClasses * inputSize parameters
% numClasses - the number of classes
% inputSize - the size N of the input vector
% lambda - penalization parameter
% data - the N x M input matrix
% labels - an M x 1 matrix containing the labels corresponding for the input data


% Unroll the parameters from beta
beta = reshape(beta, numClasses, inputSize);
numCases = size(data, 2);
groundTruth = full(sparse(labels, 1:numCases, 1));

% Calc sigmoid 
M = beta * data;
M = bsxfun(@minus, M, max(M));
p = bsxfun(@rdivide, exp(M), sum(exp(M)));

% Calc beta and cost function 
cost = -1/numCases * groundTruth(:)' * log(p(:)) + lambda/2 * sum(beta(:) .^ 2);
betagrad = -1/numCases * (groundTruth - p) * data' + lambda * beta;

% Calc gradiend
grad = [betagrad(:)];

end