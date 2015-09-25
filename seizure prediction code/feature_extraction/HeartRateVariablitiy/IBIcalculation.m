function [ibi] = IBIcalculation(R_i,sample_freq)
%% calculate ibi signal from R waves locations

t = R_i*1000/sample_freq;
ibi = [t(2:end)',diff(t)'];