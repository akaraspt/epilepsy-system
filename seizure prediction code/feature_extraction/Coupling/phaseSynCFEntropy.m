%% calculate phase Synchronisation of EEG and IBI signals, 
function [phaseEntropy,f_indx] = phaseSynCFEntropy(angle,angle2,Lf,Hf,Lf2,Hf2,sample_freq)
%% caluclate phase synchronisation via entropy method

%%input
%anple,angle2 -- row vector and the length should be the same
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
      phase_Diff(a,:) = mod((mR+nR)/(2*nR)*squeeze(angle(i,:))-(nR+mR)/(2*mR)*squeeze(angle2(j,:)),2*pi);
       f_indx(a,:) = [i,j];
    end   
end
N_feature = a;

L = sample_freq*5;
D_diff = size(phase_Diff);
n = 0;
phaseEntropy = zeros(N_feature,ceil(D_diff(2)/L));
for window = 1:L:L*(floor(D_diff(2)/L)-1)+1
    n = n+1;
    for i = 1:N_feature
%          maxmin_range = max(phase_Diff(i,window:window+L-1))-min(phase_Diff(i,window:window+L-1));
%          fd_bins = ceil(maxmin_range/(2.0*iqr(phase_Diff(i,window:window+L-1)*L^(-1/3)))); % Freedman-Diaconis 
%         scott_bins = ceil(maxmin_range/(3.5*std(signal1)*n^(-1/3))); % Scott
        sturges_bins = ceil(1+log2(L)); % Sturges
         hdat = hist(phase_Diff(i,window:window+L-1)',sturges_bins);
         hdat = hdat./repmat(sum(hdat),1,size(hdat,2));
         ent =  -sum(hdat.*log2(hdat+eps),2);
         phaseEntropy(i,n) = (log2(sturges_bins)-ent)/log2(sturges_bins);
    end
end
if mod(D_diff(2),L)>0
    n = n+1;
    window = window+L;
    for i = 1:N_feature
        maxmin_range = max(phase_Diff(i,window:end))-min(phase_Diff(i,window:end));
%         fd_bins = ceil(maxmin_range/(2.0*iqr(phase_Diff(i,window:end)*L^(-1/3)))); % Freedman-Diaconis 
%         scott_bins = ceil(maxmin_range/(3.5*std(signal1)*n^(-1/3))); % Scott
        sturges_bins = ceil(1+log2(L)); % Sturges
        hdat = hist(phase_Diff(i,window:end)',sturges_bins);
        hdat = hdat./repmat(sum(hdat),1,size(hdat,2));
        ent =  -sum(hdat.*log2(hdat+eps),2);
        phaseEntropy(i,n) = (log2(sturges_bins)-ent)/log2(sturges_bins);
    end
end
    