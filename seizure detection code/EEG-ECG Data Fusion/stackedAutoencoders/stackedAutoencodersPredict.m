function [pred, P] = stackedAutoencodersPredict(theta,...
	inputSize, hiddenSize, numClasses, netconfig, X)
                                         
	% stackedAEPredict: Takes a trained theta and a test data set,
	% and returns the predicted labels for each example.
											 
	% theta: trained weights from the autoencoder
	% inputSize: the number of input units
	% hiddenSize:  the number of hidden units *at the last hidden layer*
	% numClasses:  the number of categories
	% X: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

	% Your code should produce the prediction matrix 
	% pred, where pred(i) is argmax_c P(y(c) | x(i)).
	 
	% Unroll theta parameter
	% First extract the part which computes the softmax gradient
	softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

	% Extract out the "stack"
	stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

	% Number of hidden layers
	numHiddenLayers = numel(stack);
	
	% Last hidden layer's activations: these are the inputs of the softmax layer
	ALast = lastHiddenLayerActivations(stack, numHiddenLayers, X);

	% Class probabilities
	P = softmaxClassProbabilities(softmaxTheta, ALast);

	% predictions
	[~, pred] = max(P);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sigmoid %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [y] = sigmoid(x)
	y = 1 ./ (1 + exp(-x));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lastHiddenLayerActivations %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ALast] = lastHiddenLayerActivations(stack, numHiddenLayers, X)
	A = forwardPropagation(stack, numHiddenLayers, X);
	ALast = A{end};
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% forwardPropagation %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [A] = forwardPropagation(stack, numHiddenLayers, X)
	% Returns a cell array A of activation matrices (for all training examples)
	A = cell(numHiddenLayers+1, 1);
	A{1} = X;
	for d=1:numHiddenLayers
		ZThis = stack{d}.W * A{d};
		ZThis = bsxfun(@plus, ZThis, stack{d}.b);
		A{d+1} = sigmoid(ZThis);
	end
end