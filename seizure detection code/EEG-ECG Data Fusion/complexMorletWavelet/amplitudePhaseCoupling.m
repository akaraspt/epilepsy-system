function [a] = amplitudePhaseCoupling(powers, angles)
	% Input:
	% - powers: The power of signal 1: rows:epochs x cols:power in time 
	% - angles: The phase angles of signal 2: rows:epochs x cols:angles in time
	%
	% Output:
	% c: The APC coefficient

	a = abs(mean(powers .* exp(1i*angles),2));
end