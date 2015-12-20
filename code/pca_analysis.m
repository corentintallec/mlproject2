load('train/train/train.mat')
cas = 'CNN';
iso = 'isolate_strange';
N_sample = 2000;
switch cas
    case 'HOG'
        x_cnn = train.X_hog;
        y_dat = train.y;
    case 'CNN'
        x_cnn = train.X_cnn;
        y_dat = train.y;
end
coeff = pca(x_cnn(1:N_sample,:));
x_cnn_trans = x_cnn*coeff;
scatter(x_cnn_trans(y_dat==1,1),x_cnn_trans(y_dat==1,2),20,'r')
hold on
scatter(x_cnn_trans(y_dat==2,1),x_cnn_trans(y_dat==2,2),20,'g')
scatter(x_cnn_trans(y_dat==3,1),x_cnn_trans(y_dat==3,2),20,'b')
scatter(x_cnn_trans(y_dat==4,1),x_cnn_trans(y_dat==4,2),20,'k')
hx = xlabel('First principal component');
hy = ylabel('Second principal component');
% the following code makes the plot look nice and increase font size etc.
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;

for i = 1:N_sample
    if x_cnn_trans(i,1) <= -8 && y_dat(i) == 4
        i
    end
end


% Next you should CROP PDF using pdfcrop in linux and mac. Windows - not sure of a solution.