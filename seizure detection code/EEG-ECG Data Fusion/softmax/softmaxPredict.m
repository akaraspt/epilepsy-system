function [pred, P] = softmaxPredict(softmaxModel, X)
	% OptTheta: The optimized parameter vector
	% X: the n x m (variables x test examples) input matrix
	%    each column X(:, i) is a single test example
	%
	 
	% Unroll the parameters from theta
	softmaxTheta = softmaxModel.OptTheta;  % this provides a numClasses x inputSize matrix
	pred = zeros(1, size(X, 2));
	
	% Class probabilities
	P = softmaxClassProbabilities(softmaxTheta, X);
	
	% Predictions
	[~, pred] = max(P);
end
