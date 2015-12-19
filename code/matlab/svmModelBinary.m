% Test of SVM method binary prediction
addpath(genpath('libsvm-3.21/matlab')); 

% Load features and labels of training data
clearvars;
load train/train.mat;

X_hog = train.X_hog;
X_cnn = train.X_cnn;

% Forming dataset
data = [X_hog X_cnn];
data = zscore(data);
labels = train.y;
data = double(data);
labels = double(labels);
numInst = size(data,1); % number of data points

% split training/testing
fprintf('Splitting into train/test..\n');
idx = randperm(numInst);

splitTrain = 0.6;
splitValid = 0.2;
splitTest = 0.2;
numTrain = splitTrain * numInst; 
numValid = splitTest * numInst;
numTest = splitValid * numInst;

trainData = data(idx(1:numTrain),:); 
validData = data(idx(numTrain+1:numTrain+numValid),:);
testData = data(idx(numTrain+numValid+1:end),:);

trainLabel = labels(idx(1:numTrain)); 
validLabel = labels(idx(numTrain+1:numTrain+numValid));
testLabel = labels(idx(numTrain+numValid+1:end));

trainLabel = convertBinarySVM(trainLabel);
testLabel = convertBinarySVM(testLabel);
%%
c = logspace(-5,5,20);
for i = 1:size(c,2)
%% Model train
numLabels = max(trainLabel); % number of classes
model = cell(numLabels,1);

t = 0;
formatOpt = '-t %d -c %d';
options = sprintf(formatOpt,t,c(i));

tic, model = svmtrain(double(trainLabel), trainData, '-t 0'); toc

% Model error calculation
[pred, accuracy, decision_values] = svmpredict(testLabel, testData, model);

%Error calc
acc = sum(pred == testLabel) ./ numel(testLabel);    %# accuracy
predErrTe(i) = sum(pred ~= testLabel) ./ numel(testLabel)   %# 1-0 error
predBERTe(i) = calcBER(pred,testLabel) %# BER error
end