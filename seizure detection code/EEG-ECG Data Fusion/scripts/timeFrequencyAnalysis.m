clear;

matFiles = {'chb04_08', 'chb04_28'};

% sampling rate in Hz
r = 256;

% peak frequencies
%     1  2  3   4   5   6   7   8   9
fs = [4, 6, 8, 12, 16, 24, 32, 48, 64];
% wavelet cycles per frequency
ns = [3, 4, 4, 5, 6, 7, 8, 9, 10];

power = true;
phase = true;

for trialIndex=1:numel(matFiles)
	fprintf('Trial no. %d of %d.\n', trialIndex, numel(matFiles));
	load(matFiles{trialIndex})
	disp('EEG time-frequency analysis...')
	[EEGPowers, EEGPhases] =...
		frequencyBandPowerAndPhase(EEG, fs, ns, r, power, phase);
	disp('ECG time-frequency analysis...')
	[ECGPowers, ECGPhases] =...
		frequencyBandPowerAndPhase(ECG, fs, ns, r, power, phase);
	save(matFiles{trialIndex}, 'EEG', 'ECG',...
		'EEGPowers', 'EEGPhases', 'ECGPowers', 'ECGPhases', 'y');
end
