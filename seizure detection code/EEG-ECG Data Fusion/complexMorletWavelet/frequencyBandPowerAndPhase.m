function [Powers, Phases] = frequencyBandPowerAndPhase(D, fs, ns, r, power, phase)
	% Input:
	% - D: the data (rows:epochs x columns:timesteps)
	% - fs: the frequencies (in Hz)
	% - ns: the number of cycles per frequency
	% - r: the sampling rate of the epochs (in Hz)
	% - power: boolean, compute power or not
	% - phase: boolean, compute phase or not
	%
	% Output:
	% - Powers: the frequency-band power signals
	%           3d matrix: epochs x frequency bands x timesteps
	% - Phases: the frequency-band phase signals
	%           3d matrix: epochs x frequency bands x timesteps

	% data length in timesteps
	t = size(D,2);
	% number of epochs
	numE = size(D,1);
	% number of frequencies
	numF = max(size(fs));
	% Powers: (number of epochs, number of frequency bands, timesteps)
	if power
		Powers = zeros([numE numF t]);
	else
		Powers = [];
	end
	% Phases: (number of epochs, number of frequency bands, timesteps)
	if phase
		Phases = zeros([numE numF t]);
	else
		Phases = false;
	end

	% for each epoch
	% Uses the parallel toolbox if available
	parfor s=1:numE
		% fprintf('Epoch %d of %d.\n', s, numE);
		% the signal of the epoch
		x = D(s,:);
		% matrix of convolutions (each row is a frequency)
		C = zeros(numF, t);
		% for each frequency
		for i=1:numF
			% compute the wavelet
			w = complexMorletWavelet(fs(i), ns(i), r);
			% convolution with the signal
			C(i,:) = transpose(conv(x, w, 'same'));
		end
		% power at different frequencies
		if power
			Power = zeros(numF, t);
		end
		% phase at different frequencies
		if phase
			Phase = zeros(numF, t);
		end
		% power(time)
		for i=1:numF
			if power
				Power(i,:) = C(i,:) .* conj(C(i,:));
			end
			if phase
				Phase(i,:) = angle(C(i,:));
			end
		end
		if power
			Powers(s,:,:) = Power;
		end
		if phase
			Phases(s,:,:) = Phase;
		end
	end
end
