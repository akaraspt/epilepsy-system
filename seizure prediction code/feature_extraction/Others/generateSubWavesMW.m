function [X_complex] = generateSubWavesMW(Lf,Hf,X_o,sample_freq)

fs = (Lf+Hf)/2;
%%set cycles
ns = 8*fs./(Hf-Lf);
numF = numel(fs);
% D = size(X); %dimension of the input signal
% X_o = reshape(X,D(1),D(2)*D(3));
D_o = size(X_o);
%%calculate wavelet components of each channel
% matrix of convolutions (each row is a frequency)
% for each frequency
X_complex = zeros([D_o(1),numF,D_o(2)]);
for j=1:numF
    % compute the wavelet
    w = complexMorletWavelet(fs(j),ns(j),sample_freq);
    % convolution with the signal
    for i = 1:D_o(1)
       X_complex(i,j,:) = transpose(conv(X_o(i,:), w, 'same'));
    end
end
%X_f = real(X_complex); %the filtered signal is the real value of the components see book Page 160