function [stackedAutoencoders, netconfig] = stackedAutoencodersLinearFirstTrain( ...
	inputSize, hiddenSizes, X, y, numTrain, numClasses, ...
	hyperParamsLayers, hyperParamsSoftmax, hyperParamsFinetuning, ...
	minFuncOptionsLayers, minFuncOptionsSoftmax, minFuncOptionsFinetuning);
		
	% inputSize: the number of variables in each training example
	% hiddenSizes: a cell array in which each element hiddenSizes{i} is the size of the i-th hidden layer 
	% lambda: weight decay parameter (same for every layer)
	% sparsityParam: sparsity parameter for the autoencoder layers (same for every layer)
	% beta: sparsity weight term
	% X: data matrix: number of variables x training examples (n x m)
	% y: labels vector: 1 x training examples (1 x m)
	% numClasses: number of classes
	% scalingFactor: scaling parameter used in the initialization of softmaxTheta
	% minFuncOptions: options for minFunc optimization (same for all stages)
	%
	% stackedAutoencoders should contain two parameter vectors
	% stackedAutoencoders.theta: pre-fine-tuning 
	% stackedAutoencoders.optTheta: after fine-tuning
	% so that the performance before and after can be compared
	
	numHiddenLayers = length(hiddenSizes);
	
	% get all the sizes in one cell array for cleaner code
	layerSizes = cell(numHiddenLayers+1,1);
	layerSizes{1} = inputSize;
	for i=1:numHiddenLayers
		layerSizes{i+1} = hiddenSizes{i};
	end
	
	% Cell array of autoencoders
	autoencoders = cell(numHiddenLayers, 1);
	
	% Cell array of activations per layer (*including input layer*)
	activations = cell(numHiddenLayers+1, 1);
	activations{1} = X;
	
	% Train all the autoencoders
	for d=1:numHiddenLayers
		fprintf('Training hidden layer %d.\n', d);
		% linear first layer
		if d == 1
			fprintf('First layer is linear.\n');
			autoencoders{d} = sparseAutoencoderLinearTrain(layerSizes{d}, layerSizes{d+1}, ...
				hyperParamsLayers{d}.lambda, hyperParamsLayers{d}.sparsityParam, ...
				hyperParamsLayers{d}.beta, activations{d}, minFuncOptionsLayers{d});
		else
			autoencoders{d} = sparseAutoencoderTrain(layerSizes{d}, layerSizes{d+1}, ...
				hyperParamsLayers{d}.lambda, hyperParamsLayers{d}.sparsityParam, ...
				hyperParamsLayers{d}.beta, activations{d}, minFuncOptionsLayers{d});
		end
		activations{d+1} = sparseAutoencoderFeatureExtraction( ...
			autoencoders{d}.optTheta, layerSizes{d}, layerSizes{d+1}, activations{d});
	end
	
	% Train the softmax layer
	fprintf('Training the softmax layer.\n');
	softmaxModel = softmaxTrain(layerSizes{end}, numClasses, hyperParamsSoftmax.lambda, ...
		activations{end}(:,1:numTrain), y, hyperParamsSoftmax.scalingFactor, minFuncOptionsSoftmax);

	% Get all the parameters as a stack
	stack = cell(numHiddenLayers,1);
	for d=1:numHiddenLayers
		stack{d}.W = reshape(autoencoders{d}.optTheta(1:layerSizes{d+1}*layerSizes{d}), ...
            layerSizes{d+1}, layerSizes{d});
		stack{d}.b = autoencoders{d}.optTheta( ...
			2*layerSizes{d+1}*layerSizes{d}+1:2*layerSizes{d+1}*layerSizes{d}+layerSizes{d+1});
	end
	
	% Initialize the parameters for the deep model
	[stackparams, netconfig] = stack2params(stack);
	stackedAutoencoders.theta = [ softmaxModel.OptTheta(:) ; stackparams ];
	
	% Train the deep network
	fprintf('Training the deep network.\n');
	[stackedAutoencoders.optTheta, cost] = minFunc(@(p) stackedAutoencodersCost(p, ...
		layerSizes{1}, layerSizes{end}, numClasses, netconfig, ...
		hyperParamsFinetuning.lambda, X(:,1:numTrain), y), stackedAutoencoders.theta, ...
		minFuncOptionsFinetuning);
end