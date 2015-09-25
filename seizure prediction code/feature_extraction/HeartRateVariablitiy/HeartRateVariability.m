function output = HeartRateVariability(ibi,totalL,step,win,sample_freq)
%% note: time is in ms
%% preprocess ibi signal -- remove ectopic beats and trend
[dibi,nibi,trend,art]=preprocessIBI(ibi); 

% output.ibi = dibi;

VLF=[0,0.04]; LF=[0.04,0.15]; HF=[0.15,0.5];
a = 0;
for i2 = step*1000:step*1000:totalL
   a = a + 1;
   i1 = max(0,i2-win*1000);
   idx = find(dibi(:,1)>=i1 & dibi(:,1) <= i2);
   if ~isempty(idx)
       %% time-domain analysis
        [output.ibi_mean(a),output.ibi_max(a),output.ibi_min(a),output.ibi_SDNN(a),output.ibi_RMSSD(a)] = timeDomainHRV(dibi(idx,:),[]);
       %% time-frequency domain analysis
        [output.aVLF(a),output.aLF(a),output.aHF(a),output.aTotal(a),...
            output.pVLF(a),output.pLF(a),output.pHF(a),output.nLF(a),...
            output.nHF(a),output.LFHF(a),output.peakVLF(a),output.peakLF(a),output.peakHF(a)] = ...
            timeFreqHRV(dibi(idx,:),VLF,LF,HF,sample_freq);
   else
      output.ibi_mean(a) = nan; output.ibi_max(a)= nan; output.ibi_min(a) = nan; output.ibi_SDNN(a) = nan; output.ibi_RMSSD(a) = nan;
      output.aVLF(a) = nan; output.aLF(a) = nan; output.aHF(a) = nan; output.aTotal(a) = nan; ...
      output.pVLF(a) = nan; output.pLF(a) = nan; output.pHF(a) = nan; 
      output.nLF(a) = nan; output.nHF(a) = nan; output.LFHF(a) = nan; output.peakVLF(a) = nan; output.peakLF(a) = nan; output.peakHF(a) = nan; 
   end
end           

if mod(totalL,step*1000)>0
    i2 = totalL;
    a = a + 1;
    i1 = max(0,i2-win*1000);
    idx = find(dibi(:,1)>=i1 & dibi(:,1) <= i2);
    if ~isempty(idx)
        [output.ibi_mean(a),output.ibi_max(a),output.ibi_min(a),output.ibi_SDNN(a),output.ibi_RMSSD(a)] = timeDomainHRV(dibi(idx,:),[]);
        [output.aVLF(a),output.aLF(a),output.aHF(a),output.aTotal(a),...
            output.pVLF(a),output.pLF(a),output.pHF(a),output.nLF(a),...
            output.nHF(a),output.LFHF(a),output.peakVLF(a),output.peakLF(a),output.peakHF(a)] = ...
            timeFreqHRV(dibi(idx,:),VLF,LF,HF,sample_freq);
    else
      output.ibi_mean(a) = nan; output.ibi_max(a)= nan; output.ibi_min(a) = nan; output.ibi_SDNN(a) = nan; output.ibi_RMSSD(a) = nan;
      output.aVLF(a) = nan; output.aLF(a) = nan; output.aHF(a) = nan; output.aTotal(a) = nan; ...
      output.pVLF(a) = nan; output.pLF(a) = nan; output.pHF(a) = nan; 
      output.nLF(a) = nan; output.nHF(a) = nan; output.LFHF(a) = nan; output.peakVLF(a) = nan; output.peakLF(a) = nan; output.peakHF(a) = nan; 
    end
end
    
