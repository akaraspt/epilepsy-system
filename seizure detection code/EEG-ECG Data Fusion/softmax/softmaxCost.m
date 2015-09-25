function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, X, y)
	% numClasses: The number of classes 
	% inputSize: The number of variables n in the input
	% lambda: The weight decay parameter
	% X: the n x m (variables x training examples) input matrix
	%    each column X(:, i) is a single training example
	% y: an m x 1 (training examples x 1) matrix containing the labels
	%    corresponding for the input data
	%

	% Unroll the parameters from theta
	softmaxTheta = reshape(theta, numClasses, inputSize);
	
	% m is the number of training examples
	m = size(X, 2);

	% labels as matrix
	Y = full(sparse(y, 1:m, 1));
	
	% Class probabilities
	P = softmaxClassProbabilities(softmaxTheta, X);
	
	% Calculate cost
	cost = softmaxCostFunction(Y, P, softmaxTheta, lambda, m);

	% Calculate gradient
	grad = softmaxGradFunction(Y, P, softmaxTheta, lambda, X, m);
	
	% Roll the gradient matrices into a vector for minFunc
	grad = [grad(:)];
end