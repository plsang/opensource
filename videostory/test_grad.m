
function test_grad()
    
    N = 1000;
    M = 100;
    D = 200;
    K = 50;
    
    X = randn(D, N);
    Y = randn(M, N);
    W = randn(D, K);
    A = randn(M, K);
    S = randn(K, N);
    
    lamdaA = 0.01;
    lamdaW = 0.01;
    lamdaS = 0.01;
    
    t = 1;
    x = X(:, t);
    y = Y(:, t);
    s = S(:, t);
    
    % check gradient wrt A
    funObj = @(theta, x, y, lamda) cost_A(theta, x, y, lamda);
    J_A = @(theta) funObj(theta, S(:, 1), Y(:, 1), lamdaA);
    theta = reshape(A, prod(size(A)), 1);
    numgrad = computeNumericalGradient(J_A, theta);
    [~, symgrad] = J_A(theta);
    check_diff(numgrad - symgrad);
    
    % check gradient wrt W
    funObj = @(theta, x, y, lamda) cost_W(theta, x, y, lamda);
    J_W = @(theta) funObj(theta, X(:, 1), S(:, 1), lamdaW);
    theta = reshape(W, prod(size(W)), 1);
    numgrad = computeNumericalGradient(J_W, theta);
    [~, symgrad] = J_W(theta);
    check_diff(numgrad - symgrad);
    
    % check gradient wrt s
    funObj = @(theta, x, y) cost_VS(theta, x, y, A, W, S, lamdaA, lamdaW, lamdaS);
    J_s = @(theta) funObj(theta, x, y);
    theta = s;
    numgrad = computeNumericalGradient(J_s, theta);
    [~, symgrad] = J_s(theta);
    check_diff(numgrad - symgrad);
    numgrad - symgrad
end

function check_diff(diff),
    if all(abs(diff) < 1e-4),
        fprintf('passed gradient check\n');
    else
        fprintf('gradient check failed\n');
    end

end

function [cost, grad] = cost_A(theta, s, y, lamda)
    A = reshape(theta, prod(size(theta))/size(s,1), size(s,1)); 
    n = size(s, 2);
    cost = (1.0/n) * sum(sum((y - A*s).^2)) + lamda*norm(A, 'fro')^2;
    
    grad = -2*(y - A*s)*s' + 2*lamda*A;
    grad = grad(:);
end

function [cost, grad] = cost_W(theta, x, s, lamda)
    W = reshape(theta, size(x,1), prod(size(theta))/size(x,1)); 
    n = size(x, 2);
    cost = (1.0/n) * sum(sum((s - W'*x).^2)) + lamda*norm(W, 'fro')^2;
    
    grad = -2*x*(s - W'*x)' + 2*lamda*W;
    grad = grad(:);
end

function [cost, grad] = cost_VS(theta, x, y, A, W, S, lamdaA, lamdaW, lamdaS)
    n = size(x, 2);
    s = theta;
    cost_as = (1.0/n) * sum(sum((y - A*s).^2)) + lamdaA*norm(A, 'fro')^2 + lamdaS*norm(s, 'fro')^2;
    cost_sw = (1.0/n) * sum(sum((s - W'*x).^2)) + lamdaW*norm(W, 'fro')^2;
    cost = cost_as + cost_sw;
    
    grad = 2*(s - W'*x - A'*(y - A*s)) + 2*lamdaS*s;
end

% j(w) = 1/n * sum(norm(y - w'x)) + lamda*Frob(w)
function test_grad_4()
    lamda = 0.01;
    funObj = @(theta, x, y) test_cost_4(theta, x, y, lamda);
    x = rand(10, 100);
    y = rand(5, 100);
    J = @(theta) funObj(theta, x(:, 1), y(:, 1));
    w = rand(10, 5);
    theta = reshape(w, prod(size(w)), 1);
    
    numgrad = computeNumericalGradient(J, theta);
    [~, symgrad] = J(theta);
    
    [numgrad, symgrad]
    numgrad - symgrad
end


% j(w) = 1/n * sum(norm(y - w'x))
function test_grad_3()
    funObj = @(theta, x, y) test_cost_3(theta, x, y);
    x = rand(10, 100);
    y = rand(5, 100);
    J = @(theta) funObj(theta, x(:, 1), y(:, 1));
    w = rand(10, 5);
    theta = reshape(w, prod(size(w)), 1);
    
    numgrad = computeNumericalGradient(J, theta);
    [~, symgrad] = J(theta);
    
    [numgrad, symgrad]
    numgrad - symgrad
end

% case w is a matrix
% cost function: Frobinues norm of (w'*x) squared
% its analytical derivative is 2*x*x'*w
function test_grad_2()
    funObj = @(theta, x) test_cost_2(theta, x);
    x = rand(10, 100);
    J = @(theta) funObj(theta, x);
    w = rand(10, 5);
    theta = reshape(w, prod(size(w)), 1);
    
    numgrad = computeNumericalGradient(J, theta);
    [~, symgrad] = J(theta);
    
    [numgrad, symgrad]
    %isequal(numgrad, symgrad)
    numgrad - symgrad
end

% case w is a vector
% cost function: sum(w'*x)
function test_grad_1()
    funObj = @(w, x) test_cost_1(w, x);
    x = rand(10, 100);
    J = @(w) funObj(w, x);
    w = rand(10, 1);
    numgrad = computeNumericalGradient(J, w);
    [~, symgrad] = J(w);
    [numgrad, symgrad]
    numgrad - symgrad
end

function [cost, grad] = test_cost_1(theta, x)
    cost = sum(theta'*x);
    grad = sum(x, 2);
end


function [cost, grad] = test_cost_2(theta, x)
    w = reshape(theta, size(x,1), prod(size(theta))/size(x,1)); 
    cost = norm(w'*x, 'fro');
    cost = cost*cost;
    
    a = 2*w'*x;
    grad = x*a';
    grad = grad(:);
end

function [cost, grad] = test_cost_3(theta, x, y)
    
    w = reshape(theta, size(x,1), prod(size(theta))/size(x,1)); 
    n = size(x, 2);
    cost = (1.0/n) * sum(sum((y - w'*x).^2));
    
    grad = -2*x*(y - w'*x)';
    grad = grad(:);
end

function [cost, grad] = test_cost_4(theta, x, y, lamda)
    
    w = reshape(theta, size(x,1), prod(size(theta))/size(x,1)); 
    n = size(x, 2);
    cost = (1.0/n) * sum(sum((y - w'*x).^2)) + lamda*norm(w, 'fro')^2;
    
    grad = -2*x*(y - w'*x)' + 2*lamda*w;
    grad = grad(:);
end

function numgrad = computeNumericalGradient(J, theta)

    numgrad = zeros(size(theta));
    
    epsilon = 1e-4;

    for i =1:length(numgrad)
        oldT = theta(i);
        theta(i)=oldT+epsilon;
        pos = J(theta);
        theta(i)=oldT-epsilon;
        neg = J(theta);
        numgrad(i) = (pos-neg)/(2*epsilon);
        theta(i)=oldT;
    end
    
end