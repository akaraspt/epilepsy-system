clear;

K = load('chb04_28_features.mat');
XTestAll = K.P;
yTestAll = K.y;
K = load('chb04_08_features.mat');
XTrainAll = K.P;
yTrainAll = K.y;
clear('K');

testSeizures = { [1679:1781] , [3782:3898] };

XTestAll = zScore(XTestAll)';
yTestAll = yTestAll';
XTrainAll = zScore(XTrainAll)';
yTrainAll = yTrainAll';

numRuns = 20;
load('selectedIndexes.mat');

[hiddenSizes, hyperParamsLayers, hyperParamsSoftmax, ...
hyperParamsFinetuning, minFuncOptionsLayers, minFuncOptionsSoftmax, ...
minFuncOptionsFinetuning, partialHyperParamsFinetuning, ...
partialMinFuncOptionsSoftmax, partialMinFuncOptionsFinetuning, ...
trainFirst] = hyperParameters(350);

% Initial indexes
allIndexes = { [1:9], indexes };

for i=1:numel(allIndexes)
	if i == 1
		fprintf('Without EEG-ECG synchronization:\n');
	else
		fprintf('With EEG-ECG synchronization:\n');
	end
	XTest = XTestAll(allIndexes{i},:);
	yTest = yTestAll;
	AllProbs = zeros(2, size(XTest,2));
	for k=1:numRuns
		fprintf('Run no. %d of %d for the ensemble.\n', k, numRuns);
		XTrain = XTrainAll(allIndexes{i},N(k,:));
		yTrain = yTrainAll(N(k,:));
		inputSize = size(XTrain,1);
		numClasses = size(unique(yTrain),2);
		numTrain = size(XTrain,2);
		[stackedAutoencoders, netconfig] = stackedAutoencodersLinearFirstTrain( ...
			inputSize, hiddenSizes, XTrain, yTrain, numTrain, numClasses, ...
			hyperParamsLayers, hyperParamsSoftmax, hyperParamsFinetuning, ...
			minFuncOptionsLayers, minFuncOptionsSoftmax, minFuncOptionsFinetuning);
		[~, Probs] = stackedAutoencodersPredict(...
			stackedAutoencoders.optTheta, inputSize, ...
			hiddenSizes{end}, numClasses, netconfig, XTest);
		AllProbs = AllProbs + Probs;
	end
	[~, pred] = max(AllProbs);
	[falsePositives, onsets, C] = falsePositivesAndOnsets(yTest, pred, testSeizures);
	if i == 1
		fprintf('Results without EEG-ECG synchronization:\n');
	else
		fprintf('Results with EEG-ECG synchronization:\n');
	end
	fprintf('False positives: %d out of %d.\n', falsePositives, numel(yTestAll));
	fprintf('Onset 1 detection latency: %d seconds.\n', onsets(1));
	fprintf('Onset 2 detection latency: %d seconds.\n', onsets(2));
	fprintf('Confusion matrix:\n');
	disp(C);
	disp('-------------------------');
end