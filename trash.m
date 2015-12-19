load train/train.mat;

mytrain.X_hog = train.X_hog(1:1000,:);
mytrain.X_cnn = train.X_cnn(1:1000,:);
mytrain.y = train.y(1:1000,:);