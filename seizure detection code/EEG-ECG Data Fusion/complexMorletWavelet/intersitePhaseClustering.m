function [c] = intersitePhaseClustering(angles1, angles2, m, n)
	% Input:
	% - angles1: The phase angles of signal 1: rows:epochs x cols:angles in time 
	% - angles2: The phase angles of signal 2: rows:epochs x cols:angles in time
	% - The ratio m:n is the ratio of the frequency bands for signals 1 and 2
	%   m and n should be integers
	%
	% Output:
	% c: The ISPC coefficient

	c = abs(mean(exp(1i* (m*angles1 - n*angles2)),2));
end