%% Load data. Initialise constants and parameters
% Load features and labels of training data
clearvars;
load train/train.mat;
for j = 1:1
X_hog = train.X_hog;
X_cnn = train.X_cnn;

% Forming dataset
data = [X_cnn X_hog];
data = zscore(data);
labels = train.y;
data = double(data);
labels = double(labels);
%labels = convertBinary(labels);
numInst = size(data,1); % number of data points
numLabels = max(labels); % number of classes

%%

% split training/testing
fprintf('Splitting into train/test..\n');
idx = randperm(numInst);

splitTrain = 0.6;
splitValid = 0.2;
splitTest = 0.2;
numTrain = splitTrain * numInst; 
numValid = splitValid * numInst;
numTest = splitTest * numInst;

Tr.X = data(idx(1:numTrain),:); 
validData = data(idx(numTrain+1:numTrain+numValid),:);
Te.X = data(idx(numTrain+numValid+1:end),:);
%testData = data(idx(numTrain+1:end),:);

Tr.y = labels(idx(1:numTrain)); 
validLabel = labels(idx(numTrain+1:numTrain+numValid));
Te.y = labels(idx(numTrain+numValid+1:end));
%testLabel = labels(idx(numTrain+1:end));
trainSet = Tr.y; testSet = Te.y;

%%Train logistic model
inputData = Tr.X';
labels = Tr.y;

inputSize = size(inputData,1); % Size of input vector 
numClasses = max(labels);     % Number of classes
lambda = 1; % Weight decay parameter
%lambda = logspace(-5,3,100);
%for i = 1:size(lambda,2)
% Learning parameters

options.maxIter = 100;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            inputData, labels, options);    

% Testing
inputTestData = Te.X';
inputTrainData = Tr.X';

% Test error
[classVoteTe] = softmaxPredict(softmaxModel, inputTestData);
classVoteTe = classVoteTe';
classVoteTe = convertBinary(classVoteTe);
Te.y = convertBinary(Te.y);
predErrTe(j) = sum( classVoteTe ~= Te.y ) / length(Te.y)
predBERTe(j) = calcBER(classVoteTe, Te.y)

% Train error
[classVoteTr] = softmaxPredict(softmaxModel, inputTrainData);
classVoteTr = classVoteTr';
predErrTr = sum( classVoteTr ~= Tr.y ) / length(Tr.y);
predBERTr = calcBER(classVoteTr, Tr.y);
end
%% Plot the regelurized hyperparameter fitting
fig1 = figure;
semilogx(lambda, predBERTe,'color','r');
hold on;
semilogx(lambda, predBERTr,'color','b');
hx = xlabel('lambda');
hy = ylabel('BER Error');
grid on;
xlim([1e-5,1e+3]);
set(gca,'fontsize',16,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
%set(gca,'fontsize',8,'fontname','Helvetica');
%gca.XScale = 'log';
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
legend('Test error','Train error','Location','southeast')
print -dpdf reg.pdf
