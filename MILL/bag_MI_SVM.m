function [test_bag_label, test_inst_label, test_bag_prob, test_inst_prob] = bag_MI_SVM(para, train_bags, test_bags)

global preprocess;
global temp_train_file temp_test_file temp_output_file temp_model_file libSVM_dir; 

num_train_bag = length(train_bags);
num_test_bag = length(test_bags);

%[train_instance, dummy] = bag2instance(train_bags);
%[test_instance, dummy] = bag2instance(test_bags);
train_instance = cat(1, train_bags(:).instance);
test_instance = cat(1, test_bags(:).instance);

precomputed_kernel = 1;

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
    
    if precomputed_kernel == 1,
        fprintf('\tCalculating train kernel .. \n') ;	
        train_kernel = train_instance*train_instance';
        
        fprintf('\tCalculating test kernel .. \n') ;	
        test_kernel = train_instance*[train_instance; test_instance]';
    end
    

    for i = 1: num_train_bag
        num_inst = size(train_bags(i).instance, 1);
        if train_bags(i).label == 0
            %sample_instance(idx+1 : idx+num_inst, :) = train_bags(i).instance;
            %sample_label(idx+1 : idx+num_inst) = zeros(num_inst, 1);
            sample_instance{i} = train_bags(i).instance;
            sample_label{i} = zeros(num_inst, 1);
        else
            %sample_instance(idx+1, :) = mean(train_bags(i).instance, 1);
            %sample_label(idx+1) = 1;
            sample_instance{i} = mean(train_bags(i).instance, 1);
            sample_label{i} = 1;
            num_pos_train_bag = num_pos_train_bag + 1;
        end
    end
    
    sample_instance = cat(1, sample_instance{:});
    sample_label = cat(1, sample_label{:})';
    
    if precomputed_kernel == 1,
        fprintf('\tCalculating sample train kernel .. \n') ;	
        sample_train_kernel = sample_instance*sample_instance';
        
        fprintf('\tCalculating sample test kernel .. \n') ;	
        sample_test_kernel = sample_instance*[train_instance; test_instance]';
    end
    
    num_train_inst = size(train_instance, 1);
    num_test_inst = size(test_instance, 1);

    num_neg_train_bag = num_train_bag - num_pos_train_bag;
    num_neg_train_inst = size(sample_instance, 1) - num_pos_train_bag;
    avg_num_inst = num_neg_train_inst / num_neg_train_bag;

    selection = zeros(num_pos_train_bag, 1);
    step = 1;
    past_selection(:, 1) = selection;    
    
    while 1
        new_para = sprintf(' -NegativeWeight %.10g', 1/avg_num_inst);
        if precomputed_kernel == 0,
            [all_label_predict, all_prob_predict] = LibSVM([para new_para], sample_instance, sample_label, [train_instance; test_instance], ones(num_train_inst+num_test_inst, 1));
        else
            [all_label_predict, all_prob_predict] = LibSVM_pre([para new_para], sample_train_kernel, sample_label, sample_test_kernel, ones(num_train_inst+num_test_inst, 1));
        end

        train_label_predict = all_label_predict(1 : num_train_inst);
        train_prob_predict = all_prob_predict(1 : num_train_inst);
        test_label_predict = all_label_predict(num_train_inst+1 : num_train_inst+ num_test_inst);
        test_prob_predict = all_prob_predict(num_train_inst+1 : num_train_inst+ num_test_inst);

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
        
        sample_train_kernel = train_kernel(sample_ind, sample_ind);
        sample_test_kernel = test_kernel(sample_ind, :);

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
    end
end

test_inst_label = test_label_predict;
test_inst_prob = test_prob_predict;

idx = 0;
for i=1:num_test_bag
    num_inst = size(test_bags(i).instance, 1);
    test_bag_label(i) = any(test_inst_label(idx+1 : idx+num_inst));
    test_bag_prob(i) = max(test_inst_prob(idx+1 : idx+num_inst));
    idx = idx + num_inst;
end