
%tic;
function main(run_name, max_neg)

C1Params = [1];
C2Params = [1];

train_MED12(run_name, C1Params, C2Params, max_neg);

test_MED12(run_name, C1Params, C2Params, max_neg);

