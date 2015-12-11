%% Load data. Initialise constants and parameters
% Load features and labels of training data
load train/train.mat;

fprintf('Splitting into train/test..\n');
Tr = [];
Te = [];
   
idxs = 1:size(train.X_hog,1);
idxs = randperm(max(idxs),max(idxs));
split = 0.8;
Tr.idxs = idxs(1,1:split*size(idxs,2));
Te.idxs = idxs(1,split*size(idxs,2)+1:end);

%Tr.idxs = 1:2:size(idxs,2);
Tr.X = train.X_hog(Tr.idxs,:);
Tr.X = [Tr.X train.X_cnn(Tr.idxs,:)];
Tr.y = train.y(Tr.idxs);

%Te.idxs = 2:2:size(idxs,2);
Te.X = train.X_hog(Te.idxs,:);
Te.X = [Te.X train.X_cnn(Te.idxs,:)];
Te.y = train.y(Te.idxs);

Tr.X = double(Tr.X);
Tr.y = double(Tr.y);
Te.X = double(Te.X);
Te.X = double(Te.X);

%%
inputData = Tr.X';
labels = Tr.y;

inputSize = size(inputData,1); % Size of input vector (MNIST images are 28x28)
numClasses = 4;     % Number of classes (MNIST images fall into 10 classes)
lambda = 1e-4; % Weight decay parameter
theta = 0.005 * randn(numClasses * inputSize, 1); % Randomly initialise theta

%% Learning parameters
tic
options.maxIter = 100;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            inputData, labels, options);    
toc
%% Testing
inputTestData = Te.X';
inputTrainData = Tr.X';

% Test error
[classVoteTe] = softmaxPredict(softmaxModel, inputTestData);
classVoteTe = classVoteTe';

acc = mean(Te.y(:) == classVoteTe(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);

predErr = sum( classVoteTe ~= Te.y ) / length(Te.y);
fprintf('\nTesting error: %.2f%%\n\n', mean(predErr) * 100 );

predBERErr = calcBER(classVote, Te.y);
fprintf('\nTesting error BER: %.2f%%\n\n', mean(predBERErr) * 100 );

% Train error
[classVoteTr] = softmaxPredict(softmaxModel, inputTrainData);
classVoteTr = classVoteTr';

predErr = sum( classVoteTr ~= Tr.y ) / length(Tr.y);
fprintf('\nTesting error: %.2f%%\n\n', mean(predErr) * 100 );

predBERErr = calcBER(classVoteTr, Tr.y);
fprintf('\nTesting error BER: %.2f%%\n\n', mean(predBERErr) * 100 );
beep