function [hiddenSizes, hyperParamsLayers, hyperParamsSoftmax, ...
	hyperParamsFinetuning, minFuncOptionsLayers, minFuncOptionsSoftmax, ...
	minFuncOptionsFinetuning, partialHyperParamsFinetuning, ...
	partialMinFuncOptionsSoftmax, partialMinFuncOptionsFinetuning, ...
	trainFirst] = hyperParameters(maxIterations)

	% hidden layers
	numHiddenLayers = 2;

	% neural network hyperparameters
	lambda = 1e-7;
	beta = 1.3;
	sparsityParam = 0.05;

	% optimisation options
	method = 'lbfgs';
	lbfgsDisplay = 'off';

	hiddenSizes = cell(numHiddenLayers,1);
	hyperParamsLayers = cell(numHiddenLayers, 1);
	minFuncOptionsLayers = cell(numHiddenLayers, 1);

	for d=1:numHiddenLayers
		hiddenSizes{d} = 20;
		hyperParamsLayers{d}.lambda = lambda;
		hyperParamsLayers{d}.beta = beta;
		hyperParamsLayers{d}.sparsityParam = sparsityParam;
		% L-BFGS options
		minFuncOptionsLayers{d}.Method = method;
		minFuncOptionsLayers{d}.maxIter = maxIterations;
		minFuncOptionsLayers{d}.Corr = minFuncOptionsLayers{d}.maxIter;
		minFuncOptionsLayers{d}.display = lbfgsDisplay;
	end

	% softmax
	hyperParamsSoftmax.lambda = lambda;
	hyperParamsSoftmax.scalingFactor = 0.01;
	minFuncOptionsSoftmax.Method = method;
	minFuncOptionsSoftmax.maxIter = maxIterations;
	minFuncOptionsSoftmax.Corr = minFuncOptionsSoftmax.maxIter; 
	minFuncOptionsSoftmax.display = lbfgsDisplay;

	% fine-tuning
	hyperParamsFinetuning.lambda = lambda;
	minFuncOptionsFinetuning.Method = method;
	minFuncOptionsFinetuning.maxIter = maxIterations;
	minFuncOptionsFinetuning.Corr = minFuncOptionsFinetuning.maxIter;
	minFuncOptionsFinetuning.display = lbfgsDisplay;

	% Partial layer-1 training parameters
	partialMinFuncOptionsSoftmax = false;
	partialHyperParamsFinetuning = false;
	partialMinFuncOptionsFinetuning = false;
	% Do fine-tuning in first layer during pretraining
	trainFirst = false;
end