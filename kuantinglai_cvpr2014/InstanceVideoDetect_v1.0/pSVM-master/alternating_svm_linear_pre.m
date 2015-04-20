function [ model ] = alternating_svm_linear_pre(fea, train_kernel, Bag_idx, Bag_prop, para)

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

iter = 0;
obj_pre = inf;
ifconverge = 0;
while (ifconverge == 0)
    model = optimize_w_pre(fea, train_kernel, model, para);
    %model = optimize_w(fea, model, para);
    model = optimize_y(fea, model, para.C_2/para.C, para.ep, bag_to_idx);
    obj_now = compute_obj(fea, model, para);
    fprintf('after solving y obj = %f\n', obj_now);
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


function [model] = optimize_y(fea, model, C, ep, bag_to_idx)
%% compute the change
f = fea*model.w + model.b;
y_n = -ones(size(f,1),1);
y_p = ones(size(f,1),1);

xi_n = max(1 - y_n.*f, zeros(length(f),1));
xi_p = max(1 - y_p.*f, zeros(length(f),1));

xi_flip = xi_n - xi_p;


%% now optimize each bag
for idx = 1:length(model.bag_prop)
    current_bag_idx = bag_to_idx{idx};
    tau = -length(current_bag_idx):2:length(current_bag_idx); % from all negatives to all positives
    xi_flip_current = xi_flip(current_bag_idx);
    
    %% compute the second term of the objective function
    tilda_xi = max(0, model.bag_prop(idx) - ep - tau/2/length(current_bag_idx)-0.5);
    tilda_xi_star = max(0, -model.bag_prop(idx) - ep + tau/2/length(current_bag_idx)+0.5);
    obj_second = C*model.bag_weight(idx)*(tilda_xi + tilda_xi_star);
    
    %% compute the first term of the objective function
    [xi_flip_current_sorted, xi_idx_sorted] = sort(xi_flip_current, 'descend');
    obj_decrease = [0; cumsum(xi_flip_current_sorted)];
    obj_first = sum(xi_n(current_bag_idx)) - obj_decrease;
    obj_proportion = obj_first + obj_second';
    [~,num_to_flip] = min(obj_proportion);
    num_to_flip = num_to_flip-1;
    y_n(current_bag_idx(xi_idx_sorted(1:num_to_flip))) = ones(num_to_flip,1); % flip signs of each bag
    %% record
    model.xi_2(idx) = tilda_xi(num_to_flip+1);
    model.xi_3(idx) = tilda_xi_star(num_to_flip+1);    
end
model.y = y_n;
%model.xi = max(1 - y_n.*f, zeros(length(f),1));
end


function [ model ] = optimize_w(fea, model, para)
% this is nothing more than the regular SVM
    model_new = regular_svm_wrapper_linear(sparse(fea), model.y, para);
    model.w = model_new.w;
    model.b = 0;
    %model.alp = model_new.alp;
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
    
    %cf http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f804
    if model_new.Label(1) == -1,
        model.w = -model.w;
        model.b = -model.b;
    end
        
    %test_base = train_kernel(model_new.SVs, :);
	%sub_scores = model_new.sv_coef' * test_base - model_new.rho;   
    %[y, acc, dec] = svmpredict(train_label, [(1:N)' train_kernel], model_new);		
    
end

function [objective] = compute_obj(fea, model, para)
    f = fea*model.w + model.b;
    xi = max(zeros(length(f),1), 1 - model.y.*f);
    obj_1 = para.C * sum(xi) + 0.5*model.w'* model.w;
    obj_2 = para.C_2*sum((model.xi_2+ model.xi_3)*model.bag_weight);
    objective = obj_1 + obj_2;
end