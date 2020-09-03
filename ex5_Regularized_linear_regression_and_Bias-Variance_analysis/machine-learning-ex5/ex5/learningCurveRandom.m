function [error_train, error_val] = ...
    learningCurveRandom(X, y, Xval, yval, lambda)

% ---------------------- Sample Solution ----------------------

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);


lambda0 = 0;

for i = 1:m,
	
	Xi = X(1:i,:);
	yi = y(1:i);
	Xvali = Xval(1:i,:);
	yvali = yval(1:i);

	theta = ones(size(X,2),1);

	[theta] = trainLinearReg(Xi, yi, lambda);
	
	
	error_train(i) = linearRegCostFunction(Xi, yi, theta, lambda0);

	error_val(i) = linearRegCostFunction(Xvali, yvali, theta, lambda0);


end





% -------------------------------------------------------------

% =========================================================================

end
