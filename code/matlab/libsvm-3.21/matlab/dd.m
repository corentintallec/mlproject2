training_label_vector = randi(2, 100,1);
training_instance_matrix = randi(100,100);
model = svmtrain(training_label_vector, training_instance_matrix);