function ber = calcBER(classVote,testSet)
% BER(Balanced error calcutalion function
% classVoted - predicted classes dataset
% testSet - test dataset

C = max(testSet);
ber_sum = 0.0;
for c = 1:C
    idx = find(testSet == c);
    Nc = length(idx);
    idx_error = find(testSet(idx)~= classVote(idx));
    class_error = (1/Nc)*length(idx_error);
    ber_sum = ber_sum + class_error;
end
ber = ber_sum/double(C);

end

