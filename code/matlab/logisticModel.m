%% Load data. Initialise constants and parameters
% Load features and labels of training data
clearvars;
load train/train.mat;


X_hog = train.X_hog;
X_cnn = train.X_cnn;

%X_hog = zscore(X_hog);
%X_cnn = zscore(X_cnn);

%X_hog = pcaecon(X_hog,500);
%X_cnn = pcaecon(X_cnn,6000);

%%

fprintf('Splitting into train/test..\n');
Tr = [];
Te = [];
setSeed(randi(10,1));   
idxs = 1:size(X_hog,1);
idxs = randperm(max(idxs),max(idxs));
split = 0.8;
Tr.idxs = idxs(1,1:split*size(idxs,2));
Te.idxs = idxs(1,split*size(idxs,2)+1:end);

%Tr.idxs = 1:2:size(idxs,2);
Tr.X = X_hog(Tr.idxs,:);
Tr.X = [Tr.X X_cnn(Tr.idxs,:)];
Tr.y = train.y(Tr.idxs);

%Te.idxs = 2:2:size(idxs,2);
Te.X = X_hog(Te.idxs,:);
Te.X = [Te.X X_cnn(Te.idxs,:)];
Te.y = train.y(Te.idxs);

Tr.X = double(Tr.X);
Tr.y = double(Tr.y);
Te.X = double(Te.X);
Te.X = double(Te.X);

%
inputData = Tr.X';
labels = Tr.y;

inputSize = size(inputData,1); % Size of input vector 
numClasses = 4;     % Number of classes
%lambda = 10; % Weight decay parameter
lambda = logspace(-5,3,100);
for i = 1:size(lambda,2)
% Learning parameters
tic
options.maxIter = 100;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda(i), ...
                            inputData, labels, options);    
toc
% Testing
inputTestData = Te.X';
inputTrainData = Tr.X';

% Test error
[classVoteTe] = softmaxPredict(softmaxModel, inputTestData);
classVoteTe = classVoteTe';
predErrTe = sum( classVoteTe ~= Te.y ) / length(Te.y);
predBERTe(i) = calcBER(classVoteTe, Te.y);

% Train error
[classVoteTr] = softmaxPredict(softmaxModel, inputTrainData);
classVoteTr = classVoteTr';
predErrTr = sum( classVoteTr ~= Tr.y ) / length(Tr.y);
predBERTr(i) = calcBER(classVoteTr, Tr.y);
end

%% Plot the regelurized hyperparameter fitting
fig1 = figure;
semilogx(lambda, predBERTe,'color','r','LineWidth',2);
hold on;
semilogx(lambda, predBERTr,'color','b','LineWidth',2);
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
%% Plot learning curve
trainData = Tr.X';
trainLabels = Tr.y;
testData = Te.X';
testLabels = Te.y;
options.maxIter = 100;
lambda = 10;
numClasses = 4;
inputSize = size(trainData,1); % Size of input vector 

m = size(trainData, 2); % Number of training examples
error_train = zeros(m, 1);
error_val   = zeros(m, 1);
%%
j =0; predBERTe = []; predBERTr = [];
number_train_examples = 100:100:m; 
for i = 100:100:m
   j = j + 1;
   softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            trainData(:,1:i), trainLabels(1:i,:), options);    
      
   % Train error
   [classVoteTr] = softmaxPredict(softmaxModel, trainData(:,1:i));
   classVoteTr = classVoteTr';
   predErrTr(j) = sum( classVoteTr ~= trainLabels(1:i,:) ) / length(trainLabels(1:i,:));
   predBERTr(j) = calcBER(classVoteTr, trainLabels(1:i,:));
   
   % Test error
   [classVoteTe] = softmaxPredict(softmaxModel, testData);
   classVoteTe = classVoteTe';
   predErrTe(j) = sum( classVoteTe ~= testLabels ) / length(testLabels);
   predBERTe(j) = calcBER(classVoteTe, testLabels);
  
end

%% Plot learning curve
fig2 = figure(2);
plot(number_train_examples, predBERTe,'color','r','LineWidth',2);
hold on;
plot(number_train_examples, predBERTr,'color','b','LineWidth',2);
hx = xlabel('Number of training examples');
hy = ylabel('BER Error');
set(gca,'fontsize',16,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
legend('Test error','Train error','Location','southeast')
grid on;
print -dpdf reg1.pdf