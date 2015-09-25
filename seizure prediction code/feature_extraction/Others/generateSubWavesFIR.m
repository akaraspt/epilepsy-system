function X_f = generateSubWavesFIR(Lf,Hf,X_o,sample_freq)
%%X_o -- channel x time

numF = numel(Lf);
filter_order = 1/Lf(1)*3*sample_freq; % Lf(1) is the lowest frequency
transition_width = [0.1 0.1 0.05 0.01 0.01];
F_nyquist = sample_freq/2;

for i = 1:numF
    % Using firls
    if Hf(i) >= F_nyquist
        ffrequencies   = [  0 ...
                        (1-transition_width(i))*Lf(i) ...
                        Lf(i) ...
                        F_nyquist ...
                     ] / F_nyquist;
        idealresponse  = [ 0 0 1 1];
    else
        ffrequencies   = [  0 ...
                            (1-transition_width(i))*Lf(i) ...
                            Lf(i) ...
                            Hf(i) ...
                            (1+transition_width(i))*Hf(i) ...
                            F_nyquist ...
                         ] / F_nyquist;
        idealresponse  = [ 0 0 1 1 0 0 ];
    end
    
    firls_weights = firls(filter_order,ffrequencies,idealresponse);
    firls_weights = firls_weights.*hamming(length(firls_weights))';
%     [Fz] = frequencyContent(firls_weights);
%     figure; plot(Fz);
    X_f(:,i,:) = filtfilt(firls_weights, 1, X_o')';
end

