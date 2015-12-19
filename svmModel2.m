%% Load data. Initialise constants and parameters
% Load features and labels of training data
load train/train.mat;
%train = mytrain;
tic
fprintf('Splitting into train/test..\n');
Tr = [];
Te = [];

setSeed(10);
idxs = 1:size(train.X_    hog,1);
idxs = randperm(max(idxs),max(idxs));
split = 0.8;
Tr.idxs = idxs(1,1:split*size(idxs,2));
Te.idxs = idxs(1,split*size(idxs,2)+1:end);

Tr.X = train.X_hog(Tr.idxs,:);
Tr.X = [Tr.X train.X_cnn(Tr.idxs,:)];
Tr.y = train.y(Tr.idxs);

Te.X = train.X_hog(Te.idxs,:);
Te.X = [Te.X train.X_cnn(Te.idxs,:)];
Te.y = train.y(Te.idxs);

Tr.X = double(Tr.X);
Tr.y = double(Tr.y);
Te.X = double(Te.X);
Te.X = double(Te.X);

%% Prediction
%dumLabels = dummyvar(double(Tr.y));
t = templateSVM('Standardize',1);
classifier = fitcecoc(Tr.X, Tr.y,'Learners',t);
%classifier = fitcecoc(Tr.X, Tr.y);
%% Testing
toc

% Test error
[classVoteTe] = predict(classifier, Te.X);

acc = mean(Te.y(:) == classVoteTe(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);

predErr = sum( classVoteTe ~= Te.y ) / length(Te.y);
fprintf('\nTesting error: %.2f%%\n\n', mean(predErr) * 100 );

predBERErr = calcBER(classVoteTe, Te.y);
fprintf('\nTesting error BER: %.2f%%\n\n', mean(predBERErr) * 100 );

% Train error
[classVoteTr] = predict(classifier, Tr.X);

predErr = sum( classVoteTr ~= Tr.y ) / length(Tr.y);
fprintf('\nTraining error: %.2f%%\n\n', mean(predErr) * 100 );

predBERErr = calcBER(classVoteTr, Tr.y);
fprintf('\nTraining error BER: %.2f%%\n\n', mean(predBERErr) * 100 );
beep