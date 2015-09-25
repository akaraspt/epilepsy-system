function theta = sparseAutoencoderInitializeParameters(inputSize, hiddenSize)
	% Initialize parameters randomly based on layer sizes.
	% choose weights uniformly from the interval [-r, r]
	r  = sqrt(6) / sqrt(hiddenSize+inputSize+1);
	W1 = rand(hiddenSize, inputSize) * 2 * r - r;
	W2 = rand(inputSize, hiddenSize) * 2 * r - r;
	% bias terms
	b1 = zeros(hiddenSize, 1);
	b2 = zeros(inputSize, 1);
	% Convert weights and bias gradients to the vector form (for minFunc)
	theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];
end

