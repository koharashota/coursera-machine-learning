function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(size(X, 1), 1) X]; % 0番目の要素を足す
z2 = a1 * Theta1';
a2 = [ones(size(z2, 1), 1) sigmoid(z2)]; % 0番目の要素を足す
z3 = a2 * Theta2';
h = sigmoid(z3); % size(h) == m, num_labels の行列

% yの入力に従ってラベル分のforループを回す
for k = 1:num_labels
  y_labeled = y == k; % 任意のラベルと等しければ1、等しくないものは0が代入される
  h_labeled = h(:, k);

  J += 1 / m * ( ...
    - y_labeled' * log(h_labeled) ...
    - (1 - y_labeled)' * log(1.0 - h_labeled) ...
  );
end

% regularizeする

J += lambda / (2 * m) * ( ...
  sum(sum(Theta1 .* Theta1),2) + sum(sum(Theta2 .* Theta2),2) ...
  - sum(sum(Theta1(:,1) .* Theta1(:,1)),2) - sum(sum(Theta2(:,1) .* Theta2(:,1)),2) ...
);


%y_for_grad = zeros(size(y, 1), num_labels);
y_for_grad = zeros(num_labels, size(y, 1));

for i = 1:size(y, 1)
  for k = 1:num_labels
    if y(i, 1) == k
      y_for_grad(k, i) = 1;
    end
  end
end


Delta_3 = sigmoid(z3') - y_for_grad;

Delta_2 = Theta2' * Delta_3 .* (sigmoidGradient( [ ones(size(z2, 1), 1) z2 ]' ));
Delta_2 = Delta_2(2:end, :);


% 数式通りでない
Theta2_grad = 1 / m * Delta_3 * (a2')';
Theta1_grad = 1 / m * Delta_2 * (a1')';


% 正規化
Theta2_grad = Theta2_grad + lambda / m * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
Theta1_grad = Theta1_grad + lambda / m * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
