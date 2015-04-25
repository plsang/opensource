function [ model ] = alternating_svm_linear_pre_sim(fea, train_kernel, Bag_idx, simrank, simrank_neg, Bag_prop, para)

addpath('/net/per610a/export/das11f/plsang/tools/libsvm-3.12/matlab');

% alternating svm with projection method
% algorithm: compute y, and project y to feasible solution
% X N*d
% Bag_idx  N*1 
% Bag_prop  S*1
% fea = sparse(fea);
para.ep = 0;
%% use reuping to initialize
if ~isfield(para, 'init_y')
    para.init_y = ones(size(fea,1),1);
    r = randperm(size(fea,1));
    para.init_y(r(1:round(size(fea,1)/2))) = -1;
end

if ~isfield(para, 'max_iter')
    para.max_iter = 100;
end

model.y = para.init_y;
        
model.bag_idx = Bag_idx;
model.bag_prop = Bag_prop;
model.bag_weight = zeros(length(model.bag_prop),1);
bag_to_idx = [];
for i = 1:length(model.bag_prop)
     bag_to_idx{i}= find(model.bag_idx == i);
     %model.bag_weight(i) = length(bag_to_idx{i});
     model.bag_weight(i) = 1;
end

iter = 1;
obj_pre = inf;
ifconverge = 0;
while (ifconverge == 0)
    model = optimize_w_pre(fea, train_kernel, model, para);
    %model = optimize_w(fea, model, para);
    [obj_1, obj_2]  = compute_obj(fea, model, para, simrank, simrank_neg, iter);
    fprintf(' iter = %d, before solving y, obj_1 = %f, obj_2 = %f, obj = %f\n', iter, obj_1, obj_2, obj_1 + obj_2);
    model = optimize_y(fea, model, para, bag_to_idx, simrank, simrank_neg, iter);
    [obj_1, obj_2]  = compute_obj(fea, model, para, simrank, simrank_neg, iter);
    fprintf(' iter = %d, after solving y, obj_1 = %f, obj_2 = %f, obj = %f\n', iter, obj_1, obj_2, obj_1 + obj_2);
    obj_now = obj_1 + obj_2;
    eps = obj_pre - obj_now;
    
    if eps <= 0 || iter>=para.max_iter
    %if iter>=para.max_iter
        ifconverge = 1;
    else
        obj_pre = obj_now;
        model_pre = model;
        model_pre.obj = obj_pre;
        iter = iter+1;
    end
end
model = model_pre;
fprintf('final obj = %f\n', model.obj);
end


function [model] = optimize_y(fea, model, para, bag_to_idx, simrank, simrank_neg, iter)
    
    %% check how many proposed was accepted
    if iter > 1,
        temp_pos = (simrank == iter-1);
        conflict_idx_pre = temp_pos & (model.pre_y == -1); 
        conflict_idx = temp_pos & (model.y_ == -1); 
        fprintf('----- Solved Pos. Conflict before: %d.  Conflict after: %d.\n', length(find(conflict_idx_pre>0)), length(find(conflict_idx>0)));
        
        temp_neg = (simrank_neg == iter-1);
        conflict_idx_pre = temp_neg & (model.pre_y == 1); 
        conflict_idx = temp_neg & (model.y_ == 1); 
        fprintf('----- Solved Neg. Conflict before: %d.  Conflict after: %d.\n', length(find(conflict_idx_pre>0)), length(find(conflict_idx>0)));
    end
    
    model.pre_y = model.y;
    
    %conflict_idx = (model.pre_y ~= model.y);
    
    %conflict_01_idx = (model.y == 1) & conflict_idx;  %% only care for 0-1 change (negative -> positive)
    pre_pos = (simrank == iter);
    conflict_idx = pre_pos & (model.y == -1); 
    fprintf(' --- Pos Info. Num relevance: %d. Num conflicts: %d.\n', length(find(pre_pos>0)), length(find(conflict_idx>0)));
    model.y(pre_pos) = 1;
    
    pre_neg = (simrank_neg == iter);
    conflict_idx = pre_neg & (model.y == 1); 
    fprintf(' --- Neg Info. Num relevance: %d. Num conflicts: %d.\n', length(find(pre_neg>0)), length(find(conflict_idx>0)));
    model.y(pre_neg) = -1;
    
    % if ~isempty(find(pre_pos > 0)),
        % y(conflict_01_idx) = -1;
        % y(conflict_01_idx & pre_pos) = 1;
    % end

    %% if there is no positive instance in the positive bag,
    %% choose the one that has the highest posibility
    % for i = 1:length(model.bag_prop)
        % inst_idx = bag_to_idx{i};
        % if ~any(y(inst_idx)) && model.bag_prop(i) == 1,
            % [~, minrank_inst_idx] = min(simrank(inst_idx));
            % y(inst_idx(minrank_inst_idx)) = 1;
            % find rank of the new 
        % end
    % end
            
    %model.y = y;

end


function [ model ] = optimize_w_pre(fea, train_kernel, model, para)
% this is nothing more than the regular SVM
    svm_opts = sprintf('-t 4 -c %f -q', para.C);
    posWeight = ceil(length(find(model.y == -1))/length(find(model.y == 1)));
    weights = [+1 posWeight ; -1 1]';
    
    if 0,
        for c=[1 -1],    
            widx = find(weights(1,:)==c) ;
            svm_opts = [svm_opts sprintf(' -w%d %g', c, weights(2,widx))] ;
        end	
    end
    
    fprintf('Training with options %s .................\n', svm_opts);
    N = length(model.y);
    model_new = svmtrain(model.y, [(1:N)' train_kernel], svm_opts) ;
    model.b = -model_new.rho ;
    model.w = fea(model_new.SVs, :)'*model_new.sv_coef;
    try
        dec = model_new.sv_coef' * train_kernel(model_new.SVs, :) - model_new.rho;   
    catch
        fprintf('error');
    end
    
    model.y_ = sign(dec');
    model.dec = dec';
    
    %cf http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f804
    if model_new.Label(1) == -1,
        model.w = -model.w;
        model.b = -model.b;
        model.y = -model.y;
        model.dec = -model.dec;
    end
        
    %test_base = train_kernel(model_new.SVs, :);
	%sub_scores = model_new.sv_coef' * test_base - model_new.rho;   
    %[y, acc, dec] = svmpredict(train_label, [(1:N)' train_kernel], model_new);		
    
end

function [obj_1, obj_2] = compute_obj(fea, model, para, simrank, simrank_neg, iter)
    %f = fea*model.w + model.b;
    xi = max(zeros(length(model.dec),1), 1 - model.y.*model.dec);
    obj_1 = para.C * sum(xi) + 0.5*model.w'* model.w;
    
    % pre_all = zeros(length(simrank), 1);
    % pre_all(simrank <= iter) = 1;
    % pre_all(simrank <= iter) = -1;
    % obj_2 = para.C_2 * (sum(double(model.y ~= pre_all)));
    
    obj_2 = para.C_2 * (sum(double(model.y(simrank <= iter) ~= 1)) + sum(double(model.y(simrank_neg <= iter) ~= -1)));
    
    %obj_2 = para.C_2*sum((model.xi_2+ model.xi_3)*model.bag_weight);
    %objective = obj_1 + obj_2;
end