function activation = sparseAutoencoderFeatureExtraction(theta, inputSize, hiddenSize, X)
	% theta: trained weights from the autoencoder
	% inputSize: the number of input units
	% hiddenSize: the number of hidden units 
	% X: the n x m (variables x training examples) input matrix
	%    each column X(:, i) is a single training example

	% Exract W1 and b1 from theta as matrix and vector
	% theta has the form: [W1 W2 b1 b2]
	W1 = reshape(theta(1:hiddenSize*inputSize), hiddenSize, inputSize);
	b1 = theta(2*hiddenSize*inputSize+1:2*hiddenSize*inputSize+hiddenSize);
	
	% *half* forward propagation (layer 2 activations)
	[activation] = halfForwardPropagation(W1, b1, X);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sigmoid %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [y] = sigmoid(x)
	y = 1 ./ (1 + exp(-x));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% halfForwardPropagation %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [A2] = halfForwardPropagation(W1, b1, X)
	% Returns an activation matrix for the hidden layer (one column per training example)
	ZThis = W1 * X; 
	ZThis = bsxfun(@plus, ZThis, b1);
	A2 = sigmoid(ZThis);
end