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
N = balancedClassBatches(yTrainAll, numRuns);

[hiddenSizes, hyperParamsLayers, hyperParamsSoftmax, ...
hyperParamsFinetuning, minFuncOptionsLayers, minFuncOptionsSoftmax, ...
minFuncOptionsFinetuning, partialHyperParamsFinetuning, ...
partialMinFuncOptionsSoftmax, partialMinFuncOptionsFinetuning, ...
trainFirst] = hyperParameters(350);

% Initial indexes
indexes = [1:9];
baselineFalsePositives = inf;
baselineOnset1 = inf;
baselineOnset2 = inf;
currentFalsePositives = inf;
currentOnset1 = inf;
currentOnset2 = inf;
% Select the top 3 EEG-ECG features
numExtraFeatures = 3;
for t=1:numExtraFeatures
	if t > 1 & ~currentRoundBestFeatureIndex
		disp('*** No better additional feature found.')
		break;
	end
	currentRoundBestFeatureIndex = false;
	fprintf('Feature selection: extra feature no. %d of %d.\n',...
		t, numExtraFeatures);
	totalFeatures = 253;
	for i=10:totalFeatures
		if any(indexes == i)
			continue;
		end
		% Initial run, to get the baseline false positives 
		if t == 1 & i == 10
			fprintf('Running baseline case.\n');
			testIndexes = indexes;
		elseif t > 1 & i == 10
			continue;
		else
			fprintf('*** %d: Testing sync feature no. %d of %d.\n', t, i, totalFeatures);
			testIndexes = [indexes i];
		end
		XTest = XTestAll(testIndexes,:);
		yTest = yTestAll;
		AllProbs = zeros(2, size(XTest,2));
		tic()
		for k=1:numRuns
			fprintf('Run no. %d of %d for the ensemble.\n', k, numRuns);
			XTrain = XTrainAll(testIndexes,N(k,:));
			yTrain = yTrainAll(N(k,:));
			inputSize = size(XTrain,1);
			numClasses = numel(unique(yTrain));
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
		toc()
		[~, pred] = max(AllProbs);
		[falsePositives, onsets, ~] = falsePositivesAndOnsets(yTest, pred, testSeizures);
		if t == 1 & i == 10
			baselineFalsePositives = falsePositives;
			baselineOnset1 = onsets(1);
			baselineOnset2 = onsets(2);
			% break;
		end
		if falsePositives < currentFalsePositives
			currentFalsePositives = falsePositives;
			currentOnset1 = onsets(1);
			currentOnset2 = onsets(2);
			currentRoundBestFeatureIndex = i;
		end
	end
	if currentRoundBestFeatureIndex
		indexes = [indexes currentRoundBestFeatureIndex]
	else
		break;
	end
end

disp('Selected features:');
fs = [4, 6, 8, 12, 16, 24, 32, 48, 64];
numF = 9;
for k=1:numel(indexes(10:end))
	count = 0;
	% We have 4 sets of EEG-ECG synchronization features
	for s=1:4
		for f1=1:numF
			for f2=1:numF
				count = count + 1;
				if count == indexes(k+9)-10
					if indexes(k+9) > 10 & indexes(k+9) <= 91
						fprintf('Intersite phase clustering EEG-ECG:\n');
					elseif indexes(k+9) > 91 & indexes(k+9) <= 172
						fprintf('Phase-amplitude coupling EEG phase - ECG power:\n');
					elseif indexes(k+9) > 172 & indexes(k+9) <= 253
						fprintf('Phase-amplitude coupling EEG power - ECG phase:\n');
					elseif indexes(k+9) > 253 & indexes(k+9) <= 334
						fprintf('Power-power correlation EEG-ECG:\n');
					end
					fprintf('%d Hz - %d Hz\n', fs(f1), fs(f2));
				end
			end
		end
	end
end

save('selectedIndexes.mat', 'N', 'indexes');
