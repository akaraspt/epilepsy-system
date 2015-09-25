clear;

edfFiles = {'chb04_08', 'chb04_28'};

channels = {'FP1F7','F7T7','T7P7','P7O1','FP1F3','F3C3',...
	'C3P3','P3O1','FP2F4','F4C4','C4P4','P4O2',...
	'FP2F8','F8T8','T8P8','P8O2','FZCZ','CZPZ',...
	'P7T7','T7FT9','FT9FT10','FT10T8','T8P8','ECG'};

eegChannel = 18;
ecgChannel = 24;

seizures = { [6446:6557], [1679:1781 , 3782:3898] };

% sampling rate
r = 256;

for trialIndex=1:numel(edfFiles)
	fprintf('Trial no. %d of %d.\n', trialIndex, numel(edfFiles));
	[~, D] = edfread(strcat(edfFiles{trialIndex}, '.edf'));
	EEG = reshape(D(eegChannel,:), [r, size(D,2)/r])';
	ECG = reshape(D(ecgChannel,:), [r, size(D,2)/r])';
	% 1 is no seizure
	% 2 is seizure
	y = ones(size(EEG,1),1);
	if seizures{trialIndex}
		y(seizures{trialIndex}) = 2;
	end
	save(strcat(edfFiles{trialIndex}, '.mat'), 'EEG', 'ECG', 'y');
	clear('D', 'y');
end
