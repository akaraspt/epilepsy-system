function [softmaxModel] = softmaxTrain(inputSize, numClasses, ...
							lambda, X, y, scalingFactor, minFuncOptions)
	theta = softmaxInitializeParameters(numClasses, inputSize, scalingFactor);
	if strcmp(minFuncOptions.Method,'lbfgs')
		[optTheta, cost] = minFunc( @(p) softmaxCost(p,...
			numClasses, inputSize, lambda, X, y), theta, minFuncOptions);
		% Unroll optTheta into a matrix
		softmaxModel.OptTheta = reshape(optTheta, numClasses, inputSize);
	elseif strcmp(minFuncOptions.Method,'sgd')
		[optTheta, cost] = sgd( @(p,D,d) softmaxCost(p,...
			numClasses, inputSize, lambda, D, d), theta, minFuncOptions, X, y);
		% Unroll optTheta into a matrix
		softmaxModel.OptTheta = reshape(optTheta, numClasses, inputSize);
	end           
end                          
