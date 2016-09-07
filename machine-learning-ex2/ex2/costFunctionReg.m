function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

[J, grad] = costFunction(theta, X, y);

tmp_theta = theta;
tmp_theta(1 , 1) = 0; % theta0 が 0 の tmp_thetaを作る

J = J + lambda / ( 2 * m ) * (tmp_theta' * tmp_theta); % theta0の分をひいとく

grad_0 = grad(1, 1);

grad = grad + lambda / m * theta;

grad(1, 1) = grad_0;


% =============================================================

end
