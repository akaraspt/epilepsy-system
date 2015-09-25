function [softmaxThetaGrad] = softmaxGradFunction(Y, P, softmaxTheta, lambda, X, m)
	softmaxThetaGrad = (-1/m) * ((Y-P) * X') + (lambda * softmaxTheta);
end