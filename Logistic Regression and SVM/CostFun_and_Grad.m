function [J, G] = CostFun_and_Grad(theta, X, y, lambda)
    % CostFun_and_Grad(theta, X, y, lambda) computes the cost function and
    % gradient for logistic regression with regularization using theta as
    % the parameters, X as the features, y as the labels, and labmda as the
    % regularization parameter. 

    m = length(y);     % number of training examples
    th = theta(2:end); % vector of parameters without bias.
    % Compute cost function:
    s = sigmoid(X*theta);
    J = - 1/m * sum(y.*log(s) + (1 - y).*log(1 - s)) + lambda * (th')*th;
    % Compute gradient:
    G = 1/m * X'*(s - y) + 2*lambda*[0;th];
end