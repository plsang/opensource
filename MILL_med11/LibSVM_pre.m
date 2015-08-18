function  [Y_compute, Y_prob, model] = LibSVM_pre(para, train_kernel, Y_train, test_kernel, Y_test, model)
   
if ~exist('model', 'var'),
 
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
            
    % set up the commands, t = 4
    svm_opts = sprintf('-b 1 -t 4 %s -c %f -w1 1 -w0 %f -q', s, p(3), p(4));

    fprintf('Training with options %s .................\n', svm_opts);
    %model = svmtrain(Y_train', X_train, svm_opts);
    N = length(Y_train);
    model = svmtrain(Y_train', [(1:N)' train_kernel], svm_opts) ;
    
end

% Prediction
fprintf('Predicting..................\n');
%[new_y, acc, dec] = svmpredict(Y_test, X_test, model, '-b 1');
Nt = length(Y_test);
[new_y, acc, dec] = svmpredict(Y_test', [(1:Nt)' test_kernel'], model, '-b 1') ;		

Y_compute = new_y;
Y_prob = dec(:, model.Label==1);


