clear all;

% Load data
load train/train.mat;
img = imread( sprintf('train/imgs/train%05d.jpg', 26) );

% Plot histogram of classed distribution
clusters = train.y;

x = []; y = [];
for i = 1:4
    x = [x i];
    y = [y size(clusters(clusters == i),1)];
end

figure(1);
bar(x,y);
title('Distribution of the clusters');
xlabel('Cluster`s number');
ylabel('Images quantatity');

%% Mean and variance of clusters distribution
mean_clusters = mean(double(clusters));
std_clusters = std(double(clusters));

% Show one HOG feature
hog_feature = train.X_hog(1,:);