function [P] = softmaxClassProbabilities(Theta, X)
	% Returns a matrix of class probabilities (one column per training example) 
	% avoid overflow
	Exponent = Theta * X;
	Exponent = bsxfun(@minus, Exponent, max(Exponent, [], 1));
	ExpThetaX = exp(Exponent);
	P = bsxfun(@rdivide, ExpThetaX, sum(ExpThetaX));
end