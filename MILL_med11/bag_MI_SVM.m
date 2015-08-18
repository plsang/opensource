function [test_bag_label, test_inst_label, test_bag_prob, test_inst_prob] = bag_MI_SVM(para, train_bags, test_bags)

global preprocess;

num_train_bag = length(train_bags);
num_test_bag = length(test_bags);

train_instance = cat(1, train_bags(:).instance);
test_instance = cat(1, test_bags(:).instance);

train_label = double(cat(2, train_bags(:).inst_label));
test_label = double(cat(2, test_bags(:).inst_label));

if isempty(train_bags)

    if (~isfield(preprocess, 'model_file') || isempty(preprocess.model_file))
        error('The model file must be provided in the train_only setting!');
    end;
    eval(['!copy ' preprocess.model_file ' ' temp_model_file ]);
    [test_label_predict, test_prob_predict] = LibSVM(para, [], [], test_instance, ones(num_test_inst, 1));    
else
    %set the initial instance labels to bag labels
    idx = 0;
    num_pos_train_bag = 0;
    
    sample_instance = cell(num_train_bag);
    sample_label = cell(num_train_bag);
    
    fprintf('\tCalculating train kernel .. \n') ;	
    train_kernel = train_instance*train_instance';
    
    for i = 1: num_train_bag
        num_inst = size(train_bags(i).instance, 1);
        if train_bags(i).label == 0
            sample_instance{i} = train_bags(i).instance;
            sample_label{i} = zeros(num_inst, 1);
        else
            sample_instance{i} = mean(train_bags(i).instance, 1);
            sample_label{i} = 1;
            num_pos_train_bag = num_pos_train_bag + 1;
        end
    end
    
    sample_instance = cat(1, sample_instance{:});
    sample_label = cat(1, sample_label{:})';
    
    num_train_inst = size(train_instance, 1);
    num_test_inst = size(test_instance, 1);

    num_neg_train_bag = num_train_bag - num_pos_train_bag;
    num_neg_train_inst = size(sample_instance, 1) - num_pos_train_bag;
    avg_num_inst = num_neg_train_inst / num_neg_train_bag;
    
    fprintf('\tCalculating sample train kernel .. \n') ;	
    sample_train_kernel = sample_instance*sample_instance';
    
    fprintf('\tCalculating sample test kernel .. \n') ;	
    sample_test_kernel = sample_instance*train_instance';
    
    clear sample_instance;
    
    selection = zeros(num_pos_train_bag, 1);
    step = 1;
    past_selection(:, 1) = selection;    
    
    new_para = sprintf(' -NegativeWeight %.10g', 1/avg_num_inst);
    para = [para new_para];
    
    current_sample_ind = [];
    
    while 1,
        
        [train_label_predict, train_prob_predict, current_model] = LibSVM_pre(para, sample_train_kernel, sample_label, sample_test_kernel, train_label);
        clear sample_train_kernel sample_test_kernel;
        
        idx = 0;
        pos_idx = 1;
        
        %sample_instance = cell(num_train_bag);
        sample_label = cell(num_train_bag);
        sample_ind = zeros(num_train_inst, 1);
        sample_ptr = 0;
        
        for i=1:num_train_bag
            num_inst = size(train_bags(i).instance, 1);

            if train_bags(i).label == 0
                %sample_instance{i} = train_bags(i).instance;
                sample_label{i} = zeros(num_inst, 1);                
                this_ind = idx + [1:size(train_bags(i).instance, 1)];
                sample_ind(sample_ptr+1:sample_ptr+num_inst) = this_ind;
                
                sample_ptr = sample_ptr + num_inst;
            else
                [sort_prob, sort_idx] = sort(train_prob_predict(idx+1 : idx+num_inst));
                %sample_instance{i} = train_bags(i).instance(sort_idx(num_inst),:);
                sample_label{i} = 1;
                
                this_ind = idx + sort_idx(num_inst);
                sample_ind(sample_ptr+1) = this_ind;
                
                selection(pos_idx) = sort_idx(num_inst);
                pos_idx = pos_idx + 1;
                
                sample_ptr = sample_ptr + 1;
            end
            idx = idx + num_inst;
        end
        
        sample_ind(sample_ptr+1:end) = [];

        %sample_instance = cat(1, sample_instance{:});
        sample_label = cat(1, sample_label{:})';

        %compare the current selection with previous selection, quit if same
        difference = sum(past_selection(:, step) ~= selection);
        fprintf('Number of selection changes is %d\n', difference);
        if difference == 0, break; end;
        
        repeat_selection = 0;
        for i = 1 : step
            if all(selection == past_selection(:,i))
                repeat_selection = 1;
                break;
            end               
        end

        if repeat_selection == 1
            fprintf('Repeated training selections found, quit...\n');
            break; 
        end

        step = step + 1;
        past_selection(:, step) = selection;
        
        sample_train_kernel = train_kernel(sample_ind, sample_ind);
        sample_test_kernel = train_kernel(sample_ind, :);
        current_sample_ind = sample_ind;
    end
end

if isempty(current_sample_ind),
    warning('Only one interation!!! \n');
end

%prediction
fprintf('\tCalculating test kernel .. \n') ;	

%test_kernel = train_instance(current_sample_ind, :)*test_instance';
num_test_inst = size(test_instance, 1);
test_chunk_size = 10000;
test_chunk_ind = 1:test_chunk_size:num_test_inst;

test_inst_label = [];
test_inst_prob = [];

for ii = 1:length(test_chunk_ind),
	start_idx = test_chunk_ind(ii);
	end_idx = start_idx + test_chunk_size - 1;
	if end_idx > num_test_inst, end_idx = num_test_inst; end;
	
	cur_test_kernel = train_instance(current_sample_ind, :)*test_instance(start_idx:end_idx, :)';
	
	[cur_test_inst_label, cur_test_inst_prob] = LibSVM_pre(para, [], [], cur_test_kernel, test_label(start_idx:end_idx), current_model);
	
	test_inst_label = [test_inst_label; cur_test_inst_label];
	test_inst_prob = [test_inst_prob; cur_test_inst_prob];
end
    
%[test_inst_label, test_inst_prob] = LibSVM_pre(para, [], [], test_kernel, test_label, current_model);

idx = 0;
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

clear cur_test_kernel train_instance test_instance;