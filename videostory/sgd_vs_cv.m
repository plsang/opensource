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

function sgd_vs_cv()
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
    opt.eta0 = 0.0001;
    opt.npass = 5; % 50 epochs
    opt.k = 32;
    
    % 
    X = data';
    Y = tv';
    
    num_train = round(size(X, 2)*0.9);
    
    train_idx = randperm(size(X, 2));
    
    Xtrain = X(:, train_idx(1:num_train));
    Ytrain = Y(:, train_idx(1:num_train));
    
    opt.Xvalid = X(:, train_idx(num_train+1:end));
    opt.Yvalid = Y(:, train_idx(num_train+1:end));
    
    fprintf('cv for best eta...\n');
    best_eta = grid_search(opt);
    %train_sgd(Xtrain, Ytrain, opt);
    
end

function [best_eta, best_lamda] = grid_search(opt)
    
    X = opt.Xvalid;
    Y = opt.Yvalid;
    
    etas = logspace(-4, 0, 5); % 0.0001    0.0010    0.0100    0.1000    1.0000
    
    lamdas = logspace(-7, -3, 5); % 1e-7, 1e-6, 1e-5, 1e-4, 1e-3

    [d_x, N] = size(X);
    [d_y, ~] = size(Y);
    
    X = [X; ones(1, N) * opt.bias_term];
    
    A_ = randn(d_y, opt.k, 'single');
    W_ = randn(d_x + 1, opt.k, 'single');
    S_ = randn(opt.k, N, 'single');
    
    logfile = [mfilename('fullpath'), '.log'];
    opt.logfile = logfile;
    
    msg = sprintf('Searching for best params for A...\n');
    logmsg(logfile, msg, 1);
    [best_eta, best_lamda] = find_best_params_A(A, W, S, opt);
    msg = sprintf(' Done: best_eta = %f, best_lamda = %f...\n', best_eta, best_lamda);
    logmsg(logfile, msg, 1);
    
    msg = sprintf('Searching for best params for W...\n');
    logmsg(logfile, msg, 1);
    [best_eta, best_lamda] = find_best_params_W(A, W, S, opt);
    msg = sprintf(' Done: best_eta = %f, best_lamda = %f...\n', best_eta, best_lamda);
    logmsg(logfile, msg, 1);
    
    msg = sprintf('Searching for best params for S...\n');
    logmsg(logfile, msg, 1);
    [best_eta, best_lamda] = find_best_params_S(A, W, S, opt);
    msg = sprintf(' Done: best_eta = %f, best_lamda = %f...\n', best_eta, best_lamda);
    logmsg(logfile, msg, 1);
   
end

function [best_eta, best_lamda] = find_best_params_A(A, W, S, opt)
    
    A_original = A;
    best_eta = 0;
    best_lamda = 0;
    best_cost = Inf;
    
    for eta = etas,
        for lamda = lamdas,
            A = A_original;
            
            for epoch = 1:opt.npass,    
                
                dataIndices = randperm(N);
                
                for t = dataIndices,
                    
                    x_t = X(:, t);
                    y_t = Y(:, t); 
                    s_t = S(:, t);
                    
                    U = 2*(y_t - A*s_t);
                    
                    delta_A = -U*s_t' + 2*lamda*A;
                    
                    A = A - eta*delta_A;
                end
                
                cost = compute_cost(A, W, S, X, Y, lamda);
                
                if cost < best_cost,
                    best_cost = cost;
                    best_eta = eta;
                    best_lamda = lamda;
                end
                
                msg = sprintf(' epoch = %d; eta = %f, lamda = %f, cost = %f;  best_eta = %f, best_lamda = %f, best_cost = %f\n', epoch, eta, lamda, cost, best_eta, best_lamda, best_cost);
                
                logmsg(opt.logfile, msg, 1);
            
            end
            
            if cost == NaN,
                fprintf('NaN cost detected! Stopped\n');
                return;
            end 
        end
    end
end

function [best_eta, best_lamda] = find_best_params_W(A, W, S, opt)
    
    W_original = W;
    
    best_eta = 0;
    best_lamda = 0;
    best_cost = Inf;
    
    for eta = etas,
        for lamda = lamdas,
            W = W_original;
            
            for epoch = 1:opt.npass,            
                dataIndices = randperm(N);
            
                for t = dataIndices,
                    x_t = X(:, t);
                    y_t = Y(:, t); 
                    s_t = S(:, t);
                    
                    V = 2*(s_t - W'*x_t);
                    delta_W = -x_t*V' + 2*lamda*W;
                    W = W - eta*delta_W;
                end
                
                [~, cost] = compute_cost(A, W, S, X, Y, lamda);
                
                if cost < best_cost,
                    best_cost = cost;
                    best_eta = eta;
                    best_lamda = lamda;
                end
                
                msg = sprintf(' epoch = %d; eta = %f, lamda = %f, cost = %f;  best_eta = %f, best_lamda = %f, best_cost = %f\n', epoch, eta, lamda, cost, best_eta, best_lamda, best_cost);
                
                logmsg(opt.logfile, msg, 1);
            
            end
            
            if cost == NaN,
                fprintf('NaN cost detected! Stopped\n');
                return;
            end 
        end
    end
end

function [best_eta, best_lamda] = find_best_params_S(A, W, S, opt)
    
    S_original = S;
    best_eta = 0;
    best_lamda = 0;
    best_cost = Inf;
    
    for eta = etas,
        for lamda = lamdas,
            S = S_original;
            
            for epoch = 1:opt.npass,            
                dataIndices = randperm(N);
            
                for t = dataIndices,
                    x_t = X(:, t);
                    y_t = Y(:, t); 
                    s_t = S(:, t);
                    
                    U = 2*(y_t - A*s_t);
                    V = 2*(s_t - W'*x_t);
                    
                    delta_s = V - A'*U + 2*lamda*s_t;
                    
                    S(:, t) = s_t - eta*delta_s;
                end
                
                [~, ~, cost] = compute_cost(A, W, S, X, Y, lamda);
                
                if cost < best_cost,
                    best_cost = cost;
                    best_eta = eta;
                    best_lamda = lamda;
                end
                
                msg = sprintf(' epoch = %d; eta = %f, lamda = %f, cost = %f;  best_eta = %f, best_lamda = %f, best_cost = %f\n', epoch, eta, lamda, cost, best_eta, best_lamda, best_cost);
                
                logmsg(opt.logfile, msg, 1);
            
            end
            
            if cost == NaN,
                fprintf('NaN cost detected! Stopped\n');
                return;
            end 
        end
    end
end

function cost = compute_cost_(A, W, S, X, Y, opt)
    N = size(X, 2);
    
    L_AS = (1.0/N) * sum(sum((Y - A*S).^2)) + opt.lambda*norm(A, 'fro')^2 + opt.lambda*norm(S, 'fro')^2;
    
    L_SW = (1.0/N) * sum(sum((S - W'*X).^2)) + opt.lambda*norm(W, 'fro')^2;
    
    cost = L_AS + L_SW;
end

function [cost_as, cost_sw, cost] = compute_cost(A, W, S, X, Y, lambda)
    N = size(X, 2);
    
    cost_as = (1.0/N) * sum(sum((Y - A*S).^2)) + lambda*norm(A, 'fro')^2 + lambda*norm(S, 'fro')^2;
    
    cost_sw = (1.0/N) * sum(sum((S - W'*X).^2)) + lambda*norm(W, 'fro')^2;
    
    cost = L_AS + L_SW;
end

function cost = compute_val_cost(A, W, opt)
    
    X = opt.Xvalid;
    Y = opt.Yvalid;
    
    N = size(X, 2);
    X = [X; ones(1, N) * opt.bias_term];
    
    S = W'*X;
    
    cost = (1.0/N) * sum(sum((Y - A*S).^2));
end
    

% output
% A: 9678x1024
% S: 1024x45826
% W: 32769x1024

%X: 32768x45826
%Y: 9678x45826
function [A, W, S] = train_sgd(X, Y, opt)
    
    logfile = [mfilename('fullpath'), '.log'];
    
    [d_x, N] = size(X);
    [d_y, ~] = size(Y);

    % add bias_term as last component of the vectors (the bias is not manipulated separately)
    X = [X; ones(1, N) * opt.bias_term];
    %Xvalid = [Xvalid ; ones(1, size(Xvalid, 2)) * opt.bias_term];

    A = randn(d_y, opt.k, 'single');
    W = randn(d_x + 1, opt.k, 'single');
    S = randn(opt.k, N, 'single');
    
    % Main loop over examples. 
    % npass / n = number of epochs
    
    output_file = sprintf('/net/per610a/export/das11f/plsang/dataset/videostory/models-k%d-e%f/AW_k1024_epoch0.mat', opt.k, opt.eta0);
    output_dir = fileparts(output_file);
    if ~exist(output_dir, 'file'),
        mkdir(output_dir);
    end
    
    fprintf('saving at epoch 0 (random) ...\n');
    save(output_file, 'W', 'A', 'S', '-v7.3');
        
    best_val_cost = Inf;
    best_epoch = 0;
    
    for epoch = 1:opt.npass
        
        % in every epoch we re-shuffle data
        dataIndices = randperm(N);
        
        opt.eta = opt.eta0 / (1 + opt.lambda * opt.eta0 * epoch);
        
        it = 1;
        for t = dataIndices,
            
            if ~mod(it, 500),
                cost = compute_cost(A, W, S, X, Y, opt);
                msg = sprintf(' it=%d, cost=%.6f\n', it, cost);
                fprintf(msg);
                logmsg(logfile, msg);
            end
            
            x_t = X(:, t);
            y_t = Y(:, t); 
            s_t = S(:, t); % 1024x1
            
            U = 2*(y_t - A*s_t);
            V = 2*(s_t - W'*x_t);
            
            delta_A = -U*s_t' + 2*opt.lambda*A;
            delta_W = -x_t*V' + 2*opt.lambda*W;
            delta_s = V - A'*U + 2*opt.lambda*s_t;
            
            A = A - opt.eta*delta_A;
            W = W - opt.eta*delta_W;
            S(:, t) = s_t - opt.eta*delta_s;
            
            it = it + 1;
        end   
        
        cost = compute_cost(A, W, S, X, Y, opt);
        val_cost = compute_val_cost(A, W, opt);
        if val_cost < best_val_cost,
            best_val_cost = val_cost;
            best_epoch = epoch;
        end
        
        msg = sprintf('\n---- epoch: %d at %s, cost = %.6f, val_cost = %.6f \n', epoch, datestr(now, 'HH:MM:SS'), cost, val_cost);
        logmsg(logfile, msg);
        fprintf(msg);
        
        output_file = sprintf('/net/per610a/export/das11f/plsang/dataset/videostory/models-k%d-e%f/AW_k1024_epoch%d.mat', opt.k, opt.eta0, epoch);
        fprintf('saving at epoch %d ...\n', epoch);
        save(output_file, 'W', 'A', 'S', '-v7.3');
    end
    
end




