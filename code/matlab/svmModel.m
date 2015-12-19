% SVM Multiclass model training
addpath(genpath('libsvm-3.21/matlab')); 

% Load features and labels of training data
clearvars;
load train/train.mat;
%%
%PCA
X_hog = train.X_hog;
%X_hog = zscore(X_hog);
%X_hog = pcaecon(X_hog,2500);

X_cnn = train.X_cnn;
%X_cnn = zscore(X_cnn);
%X_cnn = pcaecon(X_cnn, min(size(X_cnn)));

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

splitTrain = 0.6;
splitValid = 0.2;
splitTest = 0.2;
numTrain = splitTrain * numInst; 
numValid = splitValid * numInst;
numTest = splitTest * numInst;

trainData = data(idx(1:numTrain),:); 
validData = data(idx(numTrain+1:numTrain+numValid),:);
testData = data(idx(numTrain+numValid+1:end),:);


trainLabel = labels(idx(1:numTrain)); 
validLabel = labels(idx(numTrain+1:numTrain+numValid));
testLabel = labels(idx(numTrain+numValid+1:end));

%%
predBERTe = [];
c = logspace(-5,5,20);
for i = 1:size(c,2)
%% Model train
tic;
model = cell(numLabels,1);
t = 0; % set type of kernel function (0-linear, 1-polynomial,2-radial,3-sigmoid)
%c = 1; % set the parameter C 
d = 3; % degree in kernel function
b = 1; % probabilty estimates
w = 1; % parameter C of class i to weight*C

formatOpt = '-t %d -c %d -b %d';
options = sprintf(formatOpt,t,c(i),b);

for k=1:numLabels
    tic, model{k} = svmtrain(double(trainLabel==k), trainData, options); toc
end
tic
% Model error calculation (with probability)
prob = zeros(numTest,numLabels);
for k=1:numLabels
    [~,~,p] = svmpredict(double(validLabel==k), validData, model{k}, '-b 1');
    prob(:,k) = p(:,model{k}.Label==1);    %# probability of class==k
end

% predict the class with the highest probability
[~,pred] = max(prob,[],2);
predErrTe(i) = sum(pred ~= validLabel) ./ numel(validLabel);   %# 1-0 error
predBERTe(i) = calcBER(pred,validLabel) %# BER error
C = confusionmat(validLabel, pred)                   %# confusion matrix
toc
toc
end

%% Plot the C parameter choising
fig1 = figure;
semilogx(c, predBERTe,'color','r','LineWidth',2);
hx = xlabel('Parameter C (SVM Method)');
hy = ylabel('BER Error');
set(gca,'fontsize',18,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1],'YLim', [0.093,0.103]);
set([hx; hy],'fontsize',14,'fontname','avantgarde','color',[.3 .3 .3]);
legend('Test error','Location','northeast')
grid on;
print -dpdf reg3.pdf



