%% Data Analysis script. Open Data. Plot distribution.
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
hx = xlabel('Cluster`s label');
hy = ylabel('Label quantatity');
set(gca,'fontsize',18,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
hold on;
grid on;
print -dpdf reg4.pdf

%% Mean and variance of clusters distribution
mean_clusters = mean(double(clusters));
std_clusters = std(double(clusters));

% Show one HOG feature
hog_feature = train.X_hog(1,:);