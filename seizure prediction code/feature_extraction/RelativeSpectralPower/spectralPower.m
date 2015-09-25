%% calculate phase Synchronisation of EEG signal, 
function [S_power,Snorm_power] = spectralPower(eeg,Lf,Hf,sample_freq)
%% calculate spectral power of electrodes of EEG signals

%% X -- channel x time

numF = numel(Lf);
L = sample_freq*5;
D_eeg = size(eeg);
X_res = [];
if mod(D_eeg(2),L)==0 
        X_in = eeg;
else
    X_in = eeg(:,1:L*floor(D_eeg(2)/L));
    X_res = eeg(:,L*floor(D_eeg(2)/L)+1:end);
end
D_in = size(X_in);
%X_w -- channel x time x window 
X_w = reshape(X_in,D_in(1),L,D_in(2)/L); %%note: the D_eeg(2)/L should be integer

power = zeros(D_eeg(1),numF,floor(D_in(2)/L));
for i = 1:D_in(2)/L
    temp = squeeze(X_w(:,:,i));
    [pxx,f] = pwelch(temp',hamming(ceil(L/8)),[],[],sample_freq);
    pxx = pxx';
    for j = 1:numF
        indx_f = find(f>=Lf(j)&f<=Hf(j));
        if ~isempty(indx_f)
            power(:,j,i) = sum(pxx(:,indx_f),2);
        else
            power(:,j,i) = 0;
        end
    end
end
% S_power = squeeze(sum(power,3));
S_power = power; % S_power -- channel x band x window

if ~isempty(X_res)
    for j = 1:numF
        L_res = mod(D_eeg(2),L);
        temp =squeeze(X_res);
        [pxx,f] = pwelch(temp',hamming(ceil(L_res/8)),[],[],sample_freq);
        pxx = pxx';
        for j = 1:numF
            indx_f = find(f>=Lf(j)&f<=Hf(j));
            if ~isempty(indx_f)
                power_res(:,j) = sum(pxx(:,indx_f),2);
            else
                power_res(:,j) = 0;
            end
        end
    end
    S_power = cat(3,S_power,power_res);
end

total = repmat(sum(S_power,2),1,numF,1);
Snorm_power = S_power./total;