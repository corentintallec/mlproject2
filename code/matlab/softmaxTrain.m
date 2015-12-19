function [softmaxModel] = softmaxTrain(inputSize, numClasses, lambda, inputData, labels, options)
% Function softmaxTrain  train a multiclass logictic regression(softmax) model
% inputSize: the size of an input vector
% numClasses: the number of classes
% lambda: regularization parameter
% inputData: an N by M matrix, M number of datapoints
% labels: M by 1 matrix
% options (optional): options
%   options.maxIter: number of iterations to train for

if ~exist('options', 'var')
    options = struct;
end

if ~isfield(options, 'maxIter')
    options.maxIter = 400;
end

% initialize parameters
beta = 0.005 * randn(numClasses * inputSize, 1);

% Use minFunc to minimize the function
addpath 'minFunc/'
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % softmaxCost.m satisfies this.
minFuncOptions.display = 'on';

[softmaxOptbeta, cost] = minFunc( @(p) softmaxCost(p, ...
                                   numClasses, inputSize, lambda, ...
                                   inputData, labels), ...
                              beta, options);

% Fold softmaxOptbeta into a nicer format
softmaxModel.optbeta = reshape(softmaxOptbeta, numClasses, inputSize);
softmaxModel.inputSize = inputSize;
softmaxModel.numClasses = numClasses;

end