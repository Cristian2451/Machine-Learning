function g = sigmoid(z)
    % Computes the sigmoid function of z
    g = 1./(1 + exp(-z));
end