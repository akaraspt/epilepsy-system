function [R_amp,R_i,delay] = RwaveDetection(ecg,sample_freq,thre,flag)
%% R waves detection based on modified pan-tompkin's algorithm


[R_amp,R_i,delay]=pan_tompkin(ecg,sample_freq,thre,0);
% [R_amp,R_i,delay]=pan_tompkinOld(ecg,sample_freq,0);

if flag == 1
f1=5; %cuttoff low frequency to get rid of baseline wander
f2=15; %cuttoff frequency to discard high frequency noise
Wn=[f1 f2]*2/sample_freq; % cutt off based on fs
N = 3; % order of 3 less processing
[a,b] = butter(N,Wn); %bandpass filtering
ecg_h = filtfilt(a,b,ecg);
figure
plot(ecg_h);
hold on,scatter(R_i,R_amp,'m')
hold off
end
% save([pwd,folderOut,num2str(file_number),'QRS.mat'],'R_amp','R_i','delay');
