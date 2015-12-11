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
    % Train a 100 hidden unit RBM and visualize its weights                            
    rand('state',0)
    sae = saesetup([size(Tr.X,2) 200]);
    sae.ae{1}.activation_function       = 'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = 0.5;
    opts.numepochs =   10;
    opts.batchsize = 100;
    
    % Make samples multiple of batchsize, so we remove some for this to be true.
    numSampToUse = opts.batchsize * floor( size(Tr.X) / opts.batchsize);
    Tr.X = Tr.X(1:numSampToUse,:);
    Tr.y = Tr.y(1:numSampToUse);
    % normalize data
    [Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std
    Te.normX = normalize(Te.X, mu, sigma);  
    % prepare labels for NN
    LL = dummyvar(Tr.y);  
    L = dummyvar(Te.y);
    
    % Train a 100 hidden unit SDAE
    sae = saetrain(sae, Tr.normX, opts);
    
    % Use the SDAE to initialize a FFNN
    nn = nnsetup([size(Tr.X,2) 200 4]);
    nn.activation_function              = 'sigm';
    nn.learningRate                     = 1;
    nn.W{1} = sae.ae{1}.W{1};

    % Train the FFNN
    opts.numepochs =   1;
    opts.batchsize = 100;
    nn = nntrain(nn, Tr.normX, LL, opts);
        
    % get the scores we need to do nnff (feed-forward)
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
