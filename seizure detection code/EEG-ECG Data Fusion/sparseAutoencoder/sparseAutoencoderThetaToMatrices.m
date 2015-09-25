function [W1, W2, b1, b2] = sparseAutoencoderThetaToMatrices(theta, inputSize, hiddenSize)
	W1 = reshape(theta(1:hiddenSize*inputSize), hiddenSize, inputSize);
	W2 = reshape(theta(hiddenSize*inputSize+1:2*hiddenSize*inputSize), inputSize, hiddenSize);
	b1 = theta(2*hiddenSize*inputSize+1:2*hiddenSize*inputSize+hiddenSize);
	b2 = theta(2*hiddenSize*inputSize+hiddenSize+1:end);
end