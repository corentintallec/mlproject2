function [pred] = softmaxPredict(softmaxModel, data)
% Function softmaxPredict calculates predictions from train 
% multiclass logistic models (softmax)
% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, M - number of dataset poins

% Get the optmized beta
beta = softmaxModel.optbeta;  % this provides a numClasses x inputSize matrix

% Calc prediction
[nop, pred] = max(beta * data);

end