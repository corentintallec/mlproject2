% Add pca and svd functions library
addpath(genpath('svdandpca/'));

% Load data
load train/train.mat;
data = train.X_cnn;
labels = train.y;

% Normalize data
data_norm = zscore(data);

%%
% Apply PCA
data_pca = pcaecon(data_norm, 6000);