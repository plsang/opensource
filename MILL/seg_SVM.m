function [test_bag_label, test_inst_label, test_bag_prob, test_inst_prob] = seg_SVM(para, train_bags, test_bags)

global preprocess;
global temp_train_file temp_test_file temp_output_file temp_model_file libSVM_dir; 

num_train_bag = length(train_bags);
num_test_bag = length(test_bags);

%set the initial instance labels to bag labels

train_instance = cat(1, train_bags(:).instance);
test_instance = cat(1, test_bags(:).instance);

train_label = double(cat(2, train_bags(:).inst_label));
test_label = double(cat(2, test_bags(:).inst_label));

num_train_inst = size(train_instance, 1);
num_test_inst = size(test_instance, 1);


fprintf('\tCalculating train kernel [seg_SVM].. \n') ;	
train_kernel = train_instance*train_instance';

[~, ~, current_model] = LibSVM_pre(para, train_kernel, train_label, train_kernel, train_label);

clear train_kernel;

%prediction
fprintf('\tCalculating test kernel .. \n') ;	
test_kernel = train_instance*test_instance';
    
[test_inst_label, test_inst_prob] = LibSVM_pre(para, [], [], test_kernel, test_label, current_model);

idx = 0;
test_bag_label = zeros(num_test_bag, 1);
for i=1:num_test_bag
    num_inst = size(test_bags(i).instance, 1);    
    test_bag_label(i) = any(test_inst_label(idx+1 : idx+num_inst));
    test_bag_prob(i) = max(test_inst_prob(idx+1 : idx+num_inst));
    idx = idx + num_inst;
end

clear test_kernel train_instance test_instance;
