function [cost,grad] = sparseAutoencoderCost(theta, inputSize, hiddenSize, ...
											lambda, sparsityParam, beta, X)

	% inputSize: the number of input units
	% hiddenSize: the number of hidden units 
	% lambda: weight decay parameter
	% sparsityParam: The desired average activation for the hidden units
	% beta: weight of sparsity penalty term
	% X: the n x m (variables x training examples) input matrix
	%    each column X(:, i) is a single training example
	  
	W1 = reshape(theta(1:hiddenSize*inputSize), hiddenSize, inputSize);
	W2 = reshape(theta(hiddenSize*inputSize+1:2*hiddenSize*inputSize), inputSize, hiddenSize);
	b1 = theta(2*hiddenSize*inputSize+1:2*hiddenSize*inputSize+hiddenSize);
	b2 = theta(2*hiddenSize*inputSize+hiddenSize+1:end);

	% Cost and gradient variables
	cost = 0;
	W1grad = zeros(size(W1)); 
	W2grad = zeros(size(W2));
	b1grad = zeros(size(b1)); 
	b2grad = zeros(size(b2));
	
	% number of training examples
	m = size(X,2);
	
	% forward propagation (layer 2 activations and predictions, i.e. layer 3 activations)
	[A2, P] = forwardPropagation(W1, W2, b1, b2, X);
	
	% Average activations for hidden layer 
	a2Avg = sum(A2, 2) ./ m;
	
	% Calculate cost
	cost = ( (1/(2*m)) * sum(columnNorms(P-X).^2) ) + ...
		( (lambda / 2) .* (sum(W1(:).^2) + sum(W2(:).^2)) ) + ...
		( beta * sumBernoulliKLdivergence(sparsityParam, a2Avg) );
	
	% Back propagation for grad calculation
	[W1grad, W2grad, b1grad, b2grad] = sparseBackPropagation(...
		W1, W2, b1, b2, X, A2, P, a2Avg, sparsityParam, beta, lambda, m);
	
	% Convert the gradients back to a vector format (for minFunc)
	grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
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
% columnNorms %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [colNorms] = columnNorms(A)  
	colNorms = sqrt(sum(A.^2,1));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sumBernoulliKLdivergence %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sumKL] = sumBernoulliKLdivergence(rho, rhoHatVec)
	rhoVec = repmat(rho, length(rhoHatVec), 1);
	onesVec = ones(size(rhoVec));
	sumKL = sum((rhoVec .* log(rhoVec ./ rhoHatVec)) + ...
		( (onesVec - rhoVec) .* ...
		log((onesVec - rhoVec) ./ (onesVec - rhoHatVec) ) ) );
end
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% forwardPropagation %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [A2, A3] = forwardPropagation(W1, W2, b1, b2, X)
	% Returns an activation matrix for the hidden layer (one column per training example)
	ZThis = W1 * X; 
	ZThis = bsxfun(@plus, ZThis, b1);
	A2 = sigmoid(ZThis);
	ZThis = W2 * A2; 
	ZThis = bsxfun(@plus, ZThis, b2);
	A3 = sigmoid(ZThis);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sparseBackPropagation %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [W1grad, W2grad, b1grad, b2grad] = sparseBackPropagation( ...
		W1, W2, b1, b2, A1, A2, A3, a2Avg, sparsityParam, beta, lambda, m)
	% sparsity terms
	sparsityParamVec = repmat(sparsityParam, length(a2Avg), 1);
	onesVec = ones(size(a2Avg));
	sparsityTerms =...
		beta .* ( -(sparsityParamVec ./ a2Avg) + ...
		((onesVec - sparsityParamVec) ./ (onesVec - a2Avg)) );
	
	% error terms	
	D3 = -(A1 - A3) .* sigmoidDerivative(A3);
	D2 = bsxfun(@plus, (W2' * D3), sparsityTerms) .* sigmoidDerivative(A2);
	
	% grad
	W1grad = (1/m) * (D2 * A1') + (lambda .* W1);
	W2grad = (1/m) * (D3 * A2') + (lambda .* W2);
	b1grad = (1/m) * sum(D2,2);
	b2grad = (1/m) * sum(D3,2);
end