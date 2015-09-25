function output = phaseD_EEG(angleEEG,Lf,Hf,sample_freq)
%f_indx channel, channel, band, band

D_eeg = size(angleEEG);

n = 0;
f_indx = [];

for i = 1:D_eeg(1)-1
    for j = i+1:D_eeg(1)
        n = n+1;
        [phase(:,n,:),f_i] = phaseSynCF(squeeze(angleEEG(i,:,:)),squeeze(angleEEG(j,:,:)),Lf,Hf,Lf,Hf,sample_freq);
        f_indx = [f_indx;repmat([i,j],size(f_i,1),1),f_i]; % channel EEG, subband EEG, subband IBI
    end
end

D_p = size(phase);
phaseD = reshape(phase,D_p(1)*D_p(2),D_p(3));
% figure;imagesc(phaseD)

output.phaseD = phaseD;
output.f_indx = f_indx;

% 
% 
% if flag == 1
%     save([pwd,folderOut,num2str(file_number),'EEGIBIphaseN.mat'],'phaseS','f_indx');
% end
