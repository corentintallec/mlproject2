clearvars;

% Load features and labels of training data
load train/train.mat;

% split half and half into train/test, use HOG features
fprintf('Splitting into train/test..\n');

Tr = [];
Te = [];

% NOTE: you should do this randomly! and k-fold!
idxs = 1:size(train.X_hog,1);
for i = 1:1
    setSeed(randi(100,1));
    idxs = randperm(max(idxs),max(idxs));

    Tr.idxs = 1:2:idxs;
    Tr.X = train.X_hog(Tr.idxs,:);
    Tr.X = [Tr.X train.X_cnn(Tr.idxs,:)];
    Tr.y = train.y(Tr.idxs);

    Te.idxs = 2:2:idxs;
    Te.X = train.X_hog(Te.idxs,:);
    Te.X = [Te.X train.X_cnn(Te.idxs,:)];
    Te.y = train.y(Te.idxs);

    %% Train Model
    %Train the SVM Classifier  
    % fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
    classifier = fitcecoc(Tr.X, Tr.y);
    classVote = predict(classifier, Te.X);

    %% Check the train error
    % get overall error [NOTE!! this is not the BER, you have to write the code
    %                    to compute the BER!]
    predErr(i) = sum( classVote ~= Te.y ) / length(Te.y);
    predBERErr(i) = calcBER(classVote, Te.y);
end

fprintf('\nTesting error: %.2f%%\n \nTesting std: %.2f%%\n', mean(predErr) * 100, std(predErr) * 100);
fprintf('\nTesting error BER: %.2f%%\n \nTesting std BER: %.2f%%\n', mean(predBERErr) * 100, std(predBERErr)* 100);
