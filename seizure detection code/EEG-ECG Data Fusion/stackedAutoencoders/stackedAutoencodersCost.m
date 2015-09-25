function [ cost, grad ] = stackedAutoencodersCost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, X, y)
                                         
	% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
	% and returns cost and gradient using a stacked autoencoder model. Used for
	% finetuning.
											 
	% theta: trained weights from the autoencoder
	% inputSize: the number of input units
	% hiddenSize: the number of hidden units *at the last hidden layer*
	% numClasses: the number of categories
	% netconfig: the network configuration of the stack
	% lambda: the weight regularization penalty
	% X: A matrix containing the training data as columns.  So, X(:,i) is the i-th training example. 
	% y: A vector containing labels, where y(i) is the label for the i-th training example

	% We first extract the part which compute the softmax gradient
	softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

	% Extract out the "stack"
	stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);
	
	% cost
	cost = 0;
	% number of training examples
	m = size(X, 2);
	% labels as matrix
	Y = full(sparse(y, 1:m, 1));

	% Number of hidden layers
	numHiddenLayers = numel(stack);

	% Get the activations per layer with forward propagation
	A = forwardPropagation(stack, numHiddenLayers, X);
	
	% Class probabilities
	P = softmaxClassProbabilities(softmaxTheta, A{end});
	
	% Calculate cost
	cost = softmaxCostFunction(Y, P, softmaxTheta, lambda, m);

	% Calculate softmaxThetaGrad
	softmaxThetaGrad = softmaxGradFunction(Y, P, softmaxTheta, lambda, A{end}, m);
	
	% Back propagation for stackgrad calculation
	stackgrad = backPropagation(stack, softmaxTheta, Y, P, A, numHiddenLayers, m);
	
	% Roll gradient vector
	grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sigmoid %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [y] = sigmoid(x)
	y = 1 ./ (1 + exp(-x));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sigmoidDerivative %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [y] = sigmoidDerivative(sigmoidValues)
	y = sigmoidValues .* (1 - sigmoidValues); 
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% errorTerms %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [D] = errorTerms(stack, softmaxTheta, Y, P, A, numHiddenLayers)
	D = cell(numHiddenLayers+1,1);
	D{end} = -(softmaxTheta' * (Y-P)) .* sigmoidDerivative(A{end});
	for d=fliplr(1:numHiddenLayers)
		D{d} = (stack{d}.W' * D{d+1}) .* sigmoidDerivative(A{d});
	end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% backPropagation %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [stackgrad] = backPropagation(stack, softmaxTheta, Y, P, A, numHiddenLayers, m)
	% Calculate the error terms per layer
	D = errorTerms(stack, softmaxTheta, Y, P, A, numHiddenLayers);
	stackgrad = cell(size(stack));
	for d=1:numHiddenLayers
		stackgrad{d}.W = D{d+1} * A{d}' ./ m;
		stackgrad{d}.b = sum(D{d+1},2) ./ m;
	end
end