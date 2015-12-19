function binaryLabels = convertBinary(inputLabels)
% Function Convert labels from 4 classes to 2 classes

m = size(inputLabels,1);
binaryLabels = zeros(m,1);
binaryLabels(find(inputLabels == 1 | inputLabels == 2 | inputLabels ==3)) = 1;
binaryLabels(find(inputLabels == 4)) = 2;

end

