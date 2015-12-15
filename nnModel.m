clearvars;
% Load features and labels of training data
load train/train.mat;

%% split half and half into train/test, use HOG features
fprintf('Splitting into train/test..\n');

Tr = [];
Te = [];
for i = 10
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


    %% Model train

    % Data preparation. Normalize data
    % dumTrainData = dummyvar(double(Tr.y));
    % [Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std
    % 
    % Mdl = mnrfit(Tr.normX, dumTrainData);
    % [~,classVote] = predict(Mdl, Te.X);

    % setup NN. The first layer needs to have number of features neurons,
    %  and the last layer the number of classes (here four).
    nn = nnsetup([size(Tr.X,2) 100 4]);
    opts.numepochs =  20;   %  Number of full sweeps through data
    opts.batchsize = 100;  %  Take a mean gradient step over this many samples

    % if == 1 => plots trainin error as the NN is trained
    %opts.plot               = 1;

    nn.learningRate = 2;

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
    i
end

fprintf('\nTesting error: %.2f%%\n\n', mean(predErr) * 100 );
fprintf('\nTesting error BER: %.2f%%\n\n', mean(predBERErr) * 100 );
