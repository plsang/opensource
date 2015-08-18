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
%test_kernel = train_instance*test_instance';

num_test_inst = size(test_instance, 1);
test_chunk_size = 10000;
test_chunk_ind = 1:test_chunk_size:num_test_inst;

test_inst_label = [];
test_inst_prob = [];

for ii = 1:length(test_chunk_ind),
	start_idx = test_chunk_ind(ii);
	end_idx = start_idx + test_chunk_size - 1;
	if end_idx > num_test_inst, end_idx = num_test_inst; end;
	
	cur_test_kernel = train_instance*test_instance(start_idx:end_idx, :)';
	
	[cur_test_inst_label, cur_test_inst_prob] = LibSVM_pre(para, [], [], cur_test_kernel, test_label(start_idx:end_idx), current_model);
	
	test_inst_label = [test_inst_label; cur_test_inst_label];
	test_inst_prob = [test_inst_prob; cur_test_inst_prob];
end
    
%[test_inst_label, test_inst_prob] = LibSVM_pre(para, [], [], test_kernel, test_label, current_model);

idx = 0;
test_bag_label = zeros(num_test_bag, 1);
for i=1:num_test_bag
    num_inst = size(test_bags(i).instance, 1);    
    if num_inst == 0,
		test_bag_label(i) = 0;
		test_bag_prob(i) = 0;
	else
		test_bag_label(i) = any(test_inst_label(idx+1 : idx+num_inst));
		test_bag_prob(i) = max(test_inst_prob(idx+1 : idx+num_inst));
	end
    idx = idx + num_inst;
end

clear cur_test_kernel train_instance test_instance;
