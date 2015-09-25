function [P] = signalPower(X, normalise)
	% Input:
	% - X: 3d matrix: epochs x signals x timesteps
	%   	or 2d matrix: epochs x timesteps
	% - normalise: boolean, normalise the power in the [0,1] interval
	%
	% Output:
	% - P: the power
	%   	if X is 3d, P is 2D
	%   	if X is 2d, P is 1D

	% Matrix dimensions
	matDims = max(size(size(X)));
	if matDims == 3
		% number of epochs
		numE = size(X,1);
		% number of signals
		numS = size(X,2);
		P = zeros(numE, numS);
		for k=1:numE
			x = squeeze(X(k,:,:))';
			P(k, :) = rmsPow(x);
		end
	elseif matDims == 2
		P = rmsPow(X')';
	end
	if normalise
		P = mapToZeroOneInterval(P);
	end
end

function [p] = rmsPow(x)
	p = rms(x) .^ 2;
end