clear;

matFiles = {'chb04_08', 'chb04_28'};

for trialIndex=1:numel(matFiles)
	fprintf('Trial no. %d of %d.\n', trialIndex, numel(matFiles));
	load(matFiles{trialIndex});

	% number of epochs
	numE = size(EEG,1);
	% number of frequencies
	numF = size(EEGPowers,2);

	% EEG features
	normalise = false;
	s = signalPower(EEG, normalise);
	% Compute the power of the power
	fprintf('Computing the EEG power of the frequency-band power...\n');
	normalise = false;
	PowersOfPowers = signalPower(EEGPowers, normalise);

	% EEG-ECG sync features
	ISPC = zeros(numE, numF^2);
	APCeegPower = zeros(numE, numF^2);
	APCecgPower = zeros(numE, numF^2);

	fprintf('Computing intersite phase clustering\n');
	fprintf('and phase-amplitude coupling between EEG and ECG...\n');
	count = 0;
	for f1=1:numF
		for f2=1:numF
			count = count + 1;
			eegAngles = squeeze(EEGPhases(:, f1, :));
			ecgAngles = squeeze(ECGPhases(:, f2, :));
			ISPC(:, count) =...
				intersitePhaseClustering(eegAngles, ecgAngles, 1, 1);

			eegPowers = squeeze(EEGPowers(:, f1, :));
			ecgAngles = squeeze(ECGPhases(:, f2, :));
			APCeegPower(:, count) =...
				amplitudePhaseCoupling(eegPowers, ecgAngles);

			ecgPowers = squeeze(ECGPowers(:, f1, :));
			eegAngles = squeeze(EEGPhases(:, f2, :));
			APCecgPower(:, count) =...
				amplitudePhaseCoupling(ecgPowers, eegAngles);
		end
	end

	P = [PowersOfPowers s ISPC APCecgPower APCeegPower];

	save(strcat(matFiles{trialIndex}, '_features.mat'), 'P', 'y');
end