function output = RelativeSpectralPower_EEG(eeg,Lf,Hf,sample_freq)
%% calculate relative spectral power of electrodes of EEG signals

% %% filter signals using a IIR 50Hz notch filter
% X_p = preprocessNotchfilter(X_o,sample_freq); % these is not a big different after filtering

%%cacluate spectral power 
[S_power,Snorm_power] = spectralPower(eeg,Lf,Hf,sample_freq);

%%calculate relative spectral power
[RS_power,RS_indx] = relativeSpectralPower(Snorm_power);

%% feature preprocessing via a smoothing approach
[smoothRS,smoothRS_norm] = featureSmoothing(RS_power);

Snorm_powerT = permute(Snorm_power,[2,1,3]);
D_T = size(Snorm_powerT);
Snorm_powerT = reshape(Snorm_powerT,D_T(1)*D_T(2),D_T(3));

output.Snorm_power = Snorm_powerT;
output.RS_power = RS_power;
output.smoothRS_norm = smoothRS_norm;
output.RS_indx = RS_indx;
S_indx = [];
for i = 1:D_T(2)
   S_indx = [S_indx;repmat([i],D_T(1),1),[1:D_T(1)]'];
end
output.S_indx =S_indx;


% save([pwd,folderOut,num2str(file_number),'RSPFIR.mat'],'RS_power','smoothRS_norm','RS_indx');
% 
% save([pwd,folderOut,num2str(file_number),'SPFIR.mat'],'S_power','Snorm_power');

