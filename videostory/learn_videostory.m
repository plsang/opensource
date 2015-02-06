%
% This script is a straightforward Matlab implementation of the SGD routines described in Perronnin CVPR 2012. 
%
% Xtrain: training examples (should be shuffled randomly on input)
% Ltrain: training labels
% opt: options structure

% input
% data        10461x32768
% labels      10461x20, for each event, there are 10 positives (1), 4991 negatives (-1) and the ramaining are labeled as 0
%

function learn_videostory
    root_dir = '/net/per610a/export/das11f/plsang/dataset/videostory';
    
    feat_file = fullfile(root_dir, 'datasets/VideoStory46K/feature_mbh.mat'); % 45826x32768
    tv_file = fullfile(root_dir, 'datasets/VideoStory46K/tv.mat');            % 45826x9678 
    dict_file = fullfile(root_dir, 'datasets/VideoStory46K/dict.mat');        % 9678x1    
    
    fprintf('loading feat file...\n');
    load(feat_file, 'data');
    
    fprintf('loading tv file...\n');
    load(tv_file);
    fprintf('loading dict file...\n');
    load(dict_file);
    
    opt = struct(); 
    %opt.eval_freq = n;  % evaluate on validation set at each epoch

    opt.lambda = 1e-6;
    opt.bias_term = 0.1;
    opt.eta = 0.00005;
    opt.npass = 50; % 50 epochs
    opt.k = 1024;
    
    % 
    X = data';
    Y = tv';
    
    num_train = round(size(X, 2)*0.75);
    
    train_idx = randperm(size(X, 2));
    
    Xtrain = X(:, train_idx(1:num_train));
    Ytrain = Y(:, train_idx(1:num_train));
    
    opt.Xvalid = X(:, train_idx(num_train+1:end));
    opt.Yvalid = Y(:, train_idx(num_train+1:end));
    
    fprintf('start training...\n');
    train_sgd(Xtrain, Ytrain, opt);
    
end

function cost = compute_cost(A, W, S, X, Y, opt)
    N = size(X, 2);
    
    L_AS = (1.0/N) * sum(sum((Y - A*S).^2)) + opt.lambda*norm(A, 'fro')^2 + opt.lambda*norm(S, 'fro')^2;
    
    L_SW = (1.0/N) * sum(sum((S - W'*X).^2)) + opt.lambda*norm(W, 'fro')^2;
    
    cost = L_AS + L_SW;
end

function cost = compute_val_cost(A, W, opt)
    
    X = opt.Xvalid;
    Y = opt.Yvalid;
    
    N = size(X, 2);
    X = [X ; ones(1, N) * opt.bias_term];
    
    S = W'*X;
    
    L_AS = (1.0/N) * sum(sum((Y - A*S).^2)) + opt.lambda*norm(A, 'fro')^2 + opt.lambda*norm(S, 'fro')^2;
    
    %L_SW = (1.0/N) * sum(sum((S - W'*X).^2)) + opt.lambda*norm(W, 'fro')^2;
    L_SW = opt.lambda*norm(W, 'fro')^2;
    
    cost = L_AS + L_SW;
end

% zero mean, unit variance
function X = normalize_matrix(X)
    X=X-mean(X(:));
    X=X/std(X(:));
end

% output
% A: 9678x1024
% S: 1024x45826
% W: 32769x1024

%X: 32768x45826
%Y: 9678x45826
function [W, A, S] = train_sgd(X, Y, opt)

[d_x, N] = size(X);
[d_y, ~] = size(Y);

if isfield(opt, 'Xvalid')
  Xvalid = opt.Xvalid;
  Yvalid = opt.Yvalid;
else
  Xvalid = [];
  Yvalid = [];
end


% add bias_term as last component of the vectors (the bias is not manipulated separately)
X = [X ; ones(1, N) * opt.bias_term];
%Xvalid = [Xvalid ; ones(1, size(Xvalid, 2)) * opt.bias_term];

d_w = d_x + 1;
d_a = d_y + 1;

%A = opt.A;
%S = opt.S;
%W = opt.W;

% A = rand(d_y, opt.k);
% W = rand(d_w, opt.k);
% S = rand(opt.k, N);

% A = normalize_matrix(A);
% S = normalize_matrix(S);
% W = normalize_matrix(W);

A = randn(d_y, opt.k, 'single');
W = randn(d_w, opt.k, 'single');
S = randn(opt.k, N, 'single');

% Main loop over examples. 
% npass / n = number of epochs
for epoch = 1:opt.npass
  
    cost = compute_cost(A, W, S, X, Y, opt);
    val_cost = compute_val_cost(A, W, opt);
    % we loop over examples, so we have to sample the classes to modify
    % (instead of looping over classes and sampling examples)
    fprintf('\n---- epoch: %d at %s, cost = %.6f, val_cost = %.6f \n', epoch, datestr(now, 'HH:MM:SS'), cost, val_cost);
    
    for t=1:N,
        if ~mod(t, 5000),
            cost = compute_cost(A, W, S, X, Y, opt);
            val_cost = compute_val_cost(A, W, opt);
            fprintf('t=%d, cost=%.6f, val_cost=%.6f\n', t, cost, val_cost);
        end
        
        %t_ = randi(N);
        x_t = X(:, t);
        y_t = Y(:, t); 
        s_t = S(:, t); % 1024x1
        
        U = 2*(y_t - A*s_t);
        V = 2*(s_t - W'*x_t);
        
        delta_A = -U*s_t' + opt.lambda*A;
        delta_W = -x_t*V' + opt.lambda*W;
        delta_S = V - A'*U + opt.lambda*s_t;
        
        A = A - opt.eta*delta_A;
        W = W - opt.eta*delta_W;
        S(:, t) = S(:, t) - opt.eta*delta_S;
    end   
    
    output_file = sprintf('/net/per610a/export/das11f/plsang/dataset/videostory/code/AW_k1024_epoch%d.mat', epoch);
    fprintf('saving at epoch %d ...\n', epoch);
    save(output_file, 'W', 'A', 'S', '-v7.3');
end

end
% output W will be used with last component of vector set to 1
% W(end, :) = W(end, :) * opt.bias_term; 


