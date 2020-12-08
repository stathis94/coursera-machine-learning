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


z = X*theta;  % m x 1
reg = (lambda/(2*m))*sum(theta(2:end).^2);
J = (1/m)*sum((-y).*log(sigmoid(z))-(1-y).*log(1-sigmoid(z))) + reg;

%grad must be (n+1)x1

% take the first column of X => from all the examples take feature 0 => equals ones(m,1)
% then make it a row vector, so now is has size 1 x m. Apply dot product
grad_0 = (1/m)*(X(:,1))'*(sigmoid(z)-y); % 1 x 1

% same but now X(:,2:end) has size n x m
grad_n = (1/m)*(X(:,2:end))'*(sigmoid(z)-y) + (lambda/m)*theta(2:end); % n x 1
grad = [grad_0 ; grad_n];



% =============================================================

end
