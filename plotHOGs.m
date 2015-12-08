% This script will plot the historgrams for the HoG features
addpath(genpath('piotr-toolbox/toolbox/'));

img = imread( sprintf('train/imgs/train%05d.jpg', 26) );

subplot(121);
imshow(img); % image itself

subplot(122);
feature = hog( single(img)/255, 17, 8);
im( hogDraw(feature) ); colormap gray;
axis off; colorbar off;