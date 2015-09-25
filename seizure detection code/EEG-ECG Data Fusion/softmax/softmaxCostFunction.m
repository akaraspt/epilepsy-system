function [cost] = softmaxCostFunction(Y, P, softmaxTheta, lambda, m)
	cost = (-1/m) * sum(sum(Y .* log(P))) + ((lambda/2) * sum(softmaxTheta(:).^2));
end