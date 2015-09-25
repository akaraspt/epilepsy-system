function output = featureExtraction(patient,folder,folderOut,filelectrode,file_number,flag)

%% load signal 
load([pwd,folder,num2str(file_number),'.mat']);

%% generate selected EEG (6 channels) and ECG signals 
[ratio,selected_C] = channel_selection(filelectrode,6,patient);
if size(elec_names,2)>1
    temp = regexp(elec_names(2:end-1),'([^ ,:]*)','tokens');
    electrodes = cat(2,temp{:});
else
    electrodes = elec_names';
end
[~,channel_Ieeg] = find(ismember(electrodes,selected_C));
eeg = signals(channel_Ieeg,:);
D_eeg = size(eeg);
[~,channel_Iecg] = find(ismember(electrodes,'ECG'));
ecg = -signals(channel_Iecg,:);
D_ecg = size(ecg);

%% Heart rate variability analysis
[R_amp,R_i,delay] = RwaveDetection(ecg,sample_freq,3,0);
if ~isempty(R_amp)
    [ibi] = IBIcalculation(R_i,sample_freq);
    step = 5; win = 60; %when win is small, very low frequency information will lose
    totalL = length(ecg)/sample_freq*1000;
    output.HRV = HeartRateVariability(ibi,totalL,step,win,sample_freq);
    output.HRV.ibi = ibi;
end

%% EEG spectral power 
% Delta 每 0.5-4Hz
% Theta 每 4-8Hz
% Alpha 每 8-15Hz 
% Beta 每 15-30Hz
% Gamma 每 30-Nquist frequency (128Hz) 
Lf = [0.5,4,8,15,30];
Hf = [4,8,15,30,128];
output.RSP = RelativeSpectralPower_EEG(eeg,Lf,Hf,sample_freq);

% %% phase synchronisation 
% %% calculate angles 
% %EEG
EEG_fW = generateSubWavesMW(Lf,Hf,eeg,sample_freq);
angleEEG = angle(EEG_fW);
powerEEG = EEG_fW.*conj(EEG_fW);
powerEEG = scaleSignal3D(powerEEG);
% 
% %IBI
% %interpolation
if ~isempty(R_amp)
    [dibi,nibi,trend,art]=preprocessIBI(ibi); 
    t_dibi = dibi(:,1);
    t_ibiR = 1/sample_freq:1/sample_freq:D_eeg(2)/sample_freq; %time values for interp in second
    ibiR = interp1(t_dibi/1000,dibi(:,2),t_ibiR,'spline'); %cubic spline interpolation  
    % LF 每 0.04 - 0.15Hz
    % HF每 0.15 - 0.5Hz
    Lf2 = [0.04,0.15]; 
    Hf2 = [0.15,0.5];
    IBI_fW = generateSubWavesMW(Lf2,Hf2,ibiR,sample_freq);
    angleIBI = angle(IBI_fW);
    powerIBI = IBI_fW.*conj(IBI_fW);
    powerIBI = scaleSignal3D(powerIBI);
end
% 
% %ECG
f1=5; %cuttoff low frequency to get rid of baseline wander
f2=15; %cuttoff frequency to discard high frequency noise
Wn=[f1 f2]*2/sample_freq; % cutt off based on fs
N = 3; % order of 3 less processing
[a,b] = butter(N,Wn); %bandpass filtering
ECG_f = filtfilt(a,b,ecg);
% ECG_f = ECG_f/ max( abs(ECG_f));
ECG_h = hilbert(ECG_f);
angleECG(1,:,:) = angle(ECG_h);
powerECG(1,:,:) = abs(ECG_h).^2;
powerECG = scaleSignal3D(powerECG);
% 
% %phaseDifference
% %EEG-EEG
% output.EEGphaseD =  phaseD_EEG(angleEEG,Lf,Hf,sample_freq);
% %EEG_ECG
% output.EEGECGphaseD = phaseD(angleEEG,angleECG,Lf,Hf,f1,f2,sample_freq);
% %EEC_IBI
% if ~isempty(R_amp)
%     output.EEGIBIphaseD = phaseD(angleEEG(:,1,:),angleIBI,Lf(1),Hf(1),Lf2,Hf2,sample_freq);
% end

%%phaseEntropy
%EEG-EEG
output.EEGphaseE = phaseE_EEG(angleEEG,Lf,Hf,sample_freq);
%EEG_ECG
output.EEGECGphaseE = phaseE(angleEEG,angleECG,Lf,Hf,f1,f2,sample_freq);
%EEC_IBI
if ~isempty(R_amp)
    output.EEGIBIphaseE = phaseE(angleEEG(:,1,:),angleIBI,Lf(1),Hf(1),Lf2,Hf2,sample_freq);
end

%%EEGpower ECGphase
%EEG-ECG
output.EEGphaseECGpower = phasepowerCoupling(angleEEG,powerECG,sample_freq);
%EEG_IBI
if ~isempty(R_amp)
    output.EEGphaseIBIpower = phasepowerCoupling(angleEEG,powerIBI,sample_freq);
end

%%EEGphase ECGpower
%EEG-ECG
output.EEGpowerECGphase = phasepowerCoupling(angleECG,powerEEG,sample_freq);
%EEG_IBI
if ~isempty(R_amp)
    output.EEGpowerIBIphase = phasepowerCoupling(angleIBI,powerEEG,sample_freq);
end

%%EEGpower ECGpower
%EEG-ECG
output.EEGpowerECGpower = powerpowerCoupling(powerECG,powerEEG,sample_freq);
%EEG_IBI
if ~isempty(R_amp)
    output.EEGpowerIBIpower = powerpowerCoupling(powerIBI,powerEEG,sample_freq);
end


if flag == 1
      save([pwd,folderOut,num2str(file_number),'Features.mat'],'output');
end



