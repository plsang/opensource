function  [Y_compute, Y_prob] = LibSVM(para, X_train, Y_train, X_test, Y_test)
   
global temp_train_file temp_test_file temp_output_file temp_model_file libSVM_dir; 

p = str2num(char(ParseParameter(para, {'-Kernel';'-KernelParam'; '-CostFactor'; '-NegativeWeight'; '-Threshold'}, {'2';'0.05';'1';'1';'0'})));

switch p(1)
    case 0
      s = '';      
    case 1
      s = sprintf('-d %.10g -g 1', p(2));
    case 2
      s = sprintf('-g %.10g', p(2));
    case 3
      s = sprintf('-r %.10g', p(2)); 
    case 4
      s = sprintf('-u "%s"', p(2));
end
        
% set up the commands
svm_opts = sprintf('-b 1 -s 0 -t %d %s -c %f -w1 1 -w0 %f -q', p(1), s, p(3), p(4));

if (~isempty(X_train)),
    fprintf('Training: .................\n');
    model = svmtrain(Y_train', X_train, svm_opts);
end;

% Prediction
fprintf('Predicting..................\n');
[new_y, acc, dec] = svmpredict(Y_test, X_test, model, '-b 1');

Y_compute = new_y;
Y_prob = dec(:, model.Label==1);


