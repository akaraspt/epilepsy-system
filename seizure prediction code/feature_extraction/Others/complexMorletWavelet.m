function [w] = complexMorletWavelet(f, n, r)
	% f: frequency (in Hz)
	% n: number of cycles
	% r: sampling rate
	
	% standard deviation of the Gaussian taper
	s = n / (2*pi*f);
	
	% scaling factor
	a = 1 / sqrt(s*sqrt(pi));
	
	% timepoints from -2 to 2 seconds
    t0 = max(1/f*5,2);
    
	t = (-t0:1/r:t0)';
	
	w = a .* exp( (-(t.^2) ./ (2*s^2)) + ((1i*2*pi*f) .* t) );
	
end