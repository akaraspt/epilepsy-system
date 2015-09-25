%% calculate phase Synchronisation of EEG and IBI signals, 
function [phaseS,f_indx] = phaseSynCF(angle,angle2,Lf,Hf,Lf2,Hf2,sample_freq)
%% caluclate phase synchronisation via phase difference

%%input
%channelA,channelB -- row vector and the length should be the same
%r -- sampling rate
%s -- window size in seconds
% is different
% the frequency bands can be changed. 

numF = size(angle,1);
numF2 = size(angle2,1);

%calculate phase difference
phase_Diff = zeros(size(angle));
a = 0;
for i = 1:numF
    for j = 1:numF2
      a = a + 1;
      mR = mean([Lf2(j),Hf2(j)]);
      nR = mean([Lf(i),Hf(i)]);
      temp_phaseD = mod((mR+nR)/(2*nR)*squeeze(angle(i,:))-(nR+mR)/(2*mR)*squeeze(angle2(j,:)),2*pi);
      phase_Diff(a,:) = exp(complex(0,temp_phaseD));
      f_indx(a,:) = [i,j];
    end   
end
N_feature = a;

L = sample_freq*5;
D_diff = size(phase_Diff);
n = 0;
phaseS = zeros(N_feature,ceil(D_diff(2)/L));
for window = 1:L:L*(floor(D_diff(2)/L)-1)+1
    n = n+1;
    for i = 1:N_feature
        phaseS(i,n) = abs(sum(phase_Diff(i,window:window+L-1)))/L;
    end
end
if mod(D_diff(2),L)>0
    n = n+1;
    window = window+L;
    for i = 1:N_feature
       phaseS(i,n) = abs(sum(phase_Diff(i,window:end)))/size(phase_Diff(i,window:end),2);
    end
end
    