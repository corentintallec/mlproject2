%% Random forest 
%% Load data. Initialise constants and parameters
% Load features and labels of training data
clearvars;
load train/train.mat;
addpath(genpath('piotr_toolbox/toolbox')); 

X_hog = train.X_hog;
X_cnn = train.X_cnn;
for i = 1:4
%% Forming dataset
data = [X_cnn X_hog];
data = zscore(data);
labels = train.y;
data = double(data);
labels = double(labels);
numInst = size(data,1); % number of data points
numLabels = max(labels); % number of classes

% split training/testing
fprintf('Splitting into train/test..\n');
idx = randperm(numInst);
%numInst = 100; % number of data points in subset
%idx = idx(1,1:numInst);

splitTrain = 0.8;
splitTest = 0.2;
numTrain = splitTrain * numInst; 
numTest = splitTest * numInst;

trainData = data(idx(1:numTrain),:); 
testData = data(idx(numTrain+1:end),:);

trainLabel = labels(idx(1:numTrain)); 
testLabel = labels(idx(numTrain+1:end));
%trainLabel = convertBinary(trainLabel);
%testLabel = convertBinary(testLabel);

%%
inputData = trainData;
labels = trainLabel;

%labels = num2cell(labels);
labels = arrayfun(@num2str, labels, 'unif', 0);
labels = cell2mat(labels);

inputSize = size(trainData,1); % Size of input vector 
numClasses = 4;     % Number of classes

%% Learning parameters
%trees = [10 100 200 500 1000 2000 3000];
%for i = 1:size(trees,2)
rng('default'); rng(1);
tic, forestModel = TreeBagger(2000,trainData, trainLabel); toc

%%
% Test error
%classVoteTe = forestApply(single(inputTestData), forestModel);
classVoteTe = predict(forestModel,testData);
classVoteTe = str2num(cell2mat(classVoteTe));
classVoteTe = convertBinary(classVoteTe);
testLabel = convertBinary(testLabel);
predErrTe = sum( classVoteTe ~= testLabel ) / length(testLabel);
predBERTe(i) = calcBER(classVoteTe, testLabel)
end
%% Plot the decision error 
figure(2);
plot(trees, predBERTe,'color','r','LineWidth',2);
hx = xlabel('Number of trees');
hy = ylabel('BER Error');
set(gca,'fontsize',16,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',16,'fontname','avantgarde','color',[.3 .3 .3]);
%legend('Test error','Train error','Location','southeast')
grid on;
print -dpdf reg2.pdf
