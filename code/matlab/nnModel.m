% Neural network Feedforward Backpropagation Neural Networks

clearvars;
% Load features and labels of training data
load train/train.mat;

%% splitting and randomizing
fprintf('Splitting into train/test..\n');

Tr = [];
Te = [];
for i = 1
    idxs = 1:size(train.X_hog,1);
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


    %% Model train

    % Data preparation. Normalize data
    addpath(genpath('DeepLearnToolbox-master/'));    
    rng(8337, 'twister');  % fix seed, this NN may be very sensitive to initialization

    % setup NN
    nn = nnsetup([size(Tr.X,2) 100 100 4]);
    nn.dropoutFraction = 0.1;   %  Dropout fraction 
    opts.numepochs =  20;   %  Number of full sweeps through data
    opts.batchsize = 100;  %  Take a mean gradient step over this many samples

    nn.learningRate = 1;

    % this neural network implementation requires number of samples to be a
    % multiple of batchsize, so we remove some for this to be true.
    numSampToUse = opts.batchsize * floor( size(Tr.X) / opts.batchsize);
    Tr.X = Tr.X(1:numSampToUse,:);
    Tr.y = Tr.y(1:numSampToUse);

    % normalize data
    [Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std

    % prepare labels for NN
    LL = [1*(Tr.y == 1), ...
          1*(Tr.y == 2), ...
          1*(Tr.y == 3), ...
          1*(Tr.y == 4) ];  % first column, p(y=1)
                            % second column, p(y=2), etc

    [nn, L] = nntrain(nn, Tr.normX, LL, opts);


    Te.normX = normalize(Te.X, mu, sigma);  % normalize test data
%%
    % to get the scores we need to do nnff (feed-forward)
    %  see for example nnpredict().
    % (This is a weird thing of this toolbox)
    nn.testing = 1;
    nn = nnff(nn, Te.normX, zeros(size(Te.normX,1), nn.size(end)));
    nn.testing = 0;

    % predict on the test set
    nnPred = nn.a{end};

    % get the most likely class
    [~,classVote] = max(nnPred,[],2);

    %% get overall error 
    predErr(i) = sum( classVote ~= Te.y ) / length(Te.y);
    predBERErr(i) = calcBER(classVote, Te.y);
    predBERTe(i) = calcBER(classVote, Te.y);
    i
end

fprintf('\nTesting error: %.2f%%\n\n', mean(predErr) * 100 );
fprintf('\nTesting error BER: %.2f%%\n\n', mean(predBERErr) * 100 );
