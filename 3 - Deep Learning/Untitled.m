X = [0 1]; y = [0.8 1]; m = 1;
lambda = 0.0; alpha = 1;

Theta1 = [0.1  0.2
          -0.1  0.1];
Theta2 = [1 -1
         1 1-0.0361];

% Input layer
a1 = X;

% Hidden layer
z2 = a1*Theta1;
a2 = sigmoid(z2);

% Output layer
z3 = a2*Theta2;
a3 = sigmoid(z3); % sigmoidGrad(z3) * a2

[~, p] = max(a3);
p = p(:);

erro = (a3 - y);
Thetas = [Theta1(:); Theta2(:)];
J = 1/2/m*sum(erro.^2) + lambda/2/m*sum(Thetas.^2); % 2*(a3 - y)/m

dZ2 = erro .* sigmoidGrad(z3);
dW2 = a2' * dZ2 + lambda/m*Theta2;
dZ1 = (dZ2 * Theta1) .* sigmoidGrad(z2);
dW1 = X' * dZ1 + lambda/m*Theta1;

grad = [dW1(:); dW2(:)];

Thetas = Thetas - alpha*grad;

function g = sigmoid(z)
g = 1.0 ./ (1.0 + exp(-z));
end

function g = sigmoidGrad(z)
g = sigmoid(z) .* (1 - sigmoid(z));
end
