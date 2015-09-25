function theta = softmaxInitializeParameters(numClasses, inputSize, scalingFactor)
	theta = scalingFactor * randn(numClasses * inputSize, 1);
end