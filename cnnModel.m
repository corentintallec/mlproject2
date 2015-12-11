clearvars;
% Load features and labels of training data
load train/train.mat;

%% split half and half into train/test, use HOG features
fprintf('Splitting into train/test..\n');

Tr = [];
Te = [];
for i = 1:1
    % NOTE: you should do this randomly! and k-fold!
    idxs = 1:size(train.X_hog,1);
    idxs = randperm(max(idxs),max(idxs));

    Tr.idxs = 1:2:size(idxs,2);
    Tr.X = train.X_hog(Tr.idxs,:);
    Tr.X = [Tr.X train.X_cnn(Tr.idxs,:)];
    Tr.y = train.y(Tr.idxs);

    Te.idxs = 2:2:size(idxs,2);
    Te.X = train.X_hog(Te.idxs,:);
    Te.X = [Te.X train.X_cnn(Te.idxs,:)];
    Te.y = train.y(Te.idxs);
    
    Tr.X = double(Tr.X);
    Tr.y = double(Tr.y);
    Te.X = double(Te.X);
    Te.y = double(Te.y);


    %% Model train

    % Data preparation. Normalize data
    % dumTrainData = dummyvar(double(Tr.y));
    % [Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std
    % 
    % Mdl = mnrfit(Tr.normX, dumTrainData);
    % [~,classVote] = predict(Mdl, Te.X);
    % Make samples multiple of batchsize, so we remove some for this to be true.
    
    
    rand('state',0)
    cnn.layers = {
        struct('type', 'i') %input layer
        struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 2) %sub sampling layer
        struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 2) %subsampling layer
    };
    opts.alpha = 1;
    opts.batchsize = 50;
    opts.numepochs = 1;
    cnn = cnnsetup(cnn, Tr.X, Tr.y);
    
    numSampToUse = opts.batchsize * floor( size(Tr.X) / opts.batchsize);
    Tr.X = Tr.X(1:numSampToUse,:);
    Tr.y = Tr.y(1:numSampToUse);
    % normalize data
    [Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std
    Te.normX = normalize(Te.X, mu, sigma);  
    % prepare labels for NN
    LL = dummyvar(Tr.y);  
    L = dummyvar(Te.y);
  
    % Train 
    cnn = cnntrain(cnn, Tr.normX, LL, opts);

    [er, classVote] = cnntest(cnn, Te.normX, L);

    %plot mean squared error
    figure; plot(cnn.rL);


    %% get overall error 
    predErr(i) = sum( classVote ~= Te.y ) / length(Te.y);
    predBERErr(i) = calcBER(classVote, Te.y);
    i
end

fprintf('\nTesting error: %.2f%%\n\n', mean(predErr) * 100 );
fprintf('\nTesting error BER: %.2f%%\n\n', mean(predBERErr) * 100 );
