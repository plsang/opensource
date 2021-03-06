function [test_bag_label, test_inst_label, test_bag_prob, test_inst_prob] = inst_MI_SVM(para, train_bags, test_bags)

global preprocess;
global temp_train_file temp_test_file temp_output_file temp_model_file libSVM_dir; 

num_train_bag = length(train_bags);
num_test_bag = length(test_bags);

%set the initial instance labels to bag labels


%%[train_instance, dummy] = bag2instance(train_bags);
%%[test_instance, dummy] = bag2instance(test_bags);

train_instance = cat(1, train_bags(:).instance);
test_instance = cat(1, test_bags(:).instance);

train_label = double(cat(2, train_bags(:).inst_label));
test_label = double(cat(2, test_bags(:).inst_label));

num_train_inst = size(train_instance, 1);
num_test_inst = size(test_instance, 1);

if isempty(train_bags)
    if (~isfield(preprocess, 'model_file') || isempty(preprocess.model_file))
        error('The model file must be provided in the train_only setting!');
    end;
    eval(['!copy ' preprocess.model_file ' ' temp_model_file ]);
    [test_label_predict, test_prob_predict] = LibSVM(para, [], [], test_instance, ones(num_test_inst, 1));    
else
    
    step = 1;
    past_train_label(step,:) = train_label;
    
    
    fprintf('\tCalculating train kernel .. \n') ;	
    train_kernel = train_instance*train_instance';
    
    while 1
        %num_pos_label = sum(train_label == 1);
        %num_neg_label = sum(train_label == 0);
        %new_para = sprintf(' -NegativeWeight %.10g', (num_pos_label / num_neg_label));
        
        [train_label_predict, train_prob_predict, current_model] = LibSVM_pre(para, train_kernel, train_label, train_kernel, train_label);

        idx = 0;
        for i=1:num_train_bag
            num_inst = size(train_bags(i).instance, 1);

            if train_bags(i).label == 0
                train_label(idx+1 : idx+num_inst) = zeros(num_inst, 1);
            else
                if any(train_label_predict(idx+1 : idx+num_inst))
                    train_label(idx+1 : idx+num_inst) = train_label_predict(idx+1 : idx+num_inst);
                else
                    [sort_prob, sort_idx] = sort(train_prob_predict(idx+1 : idx+num_inst));
                    train_label(idx+1 : idx+num_inst) = zeros(num_inst, 1);
                    train_label(idx + sort_idx(num_inst)) = 1;
                end
            end
            idx = idx + num_inst;
        end
        
        difference = sum(past_train_label(step,:) ~= train_label);
        fprintf('Number of label changes is %d\n', difference);
        if difference == 0, break; end;
         
        repeat_label = 0;
        for i = 1 : step
            if all(train_label == past_train_label(i, :))
                repeat_label = 1;
                break;
            end               
        end

        if repeat_label == 1
            fprintf('Repeated training labels found, quit...\n');
            break; 
        end

        step = step + 1;
        past_train_label(step, :) = train_label;
         
    end    
end

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
		%test_bag_prob(i) = max(test_inst_prob(idx+1 : idx+num_inst));
		test_bag_prob(i) = mean(test_inst_prob(idx+1 : idx+num_inst));
	end
    idx = idx + num_inst;
end

clear test_kernel train_instance test_instance;