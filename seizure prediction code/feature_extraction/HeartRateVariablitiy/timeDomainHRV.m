
function [ibi_mean,ibi_max,ibi_min,ibi_SDNN,ibi_RMSSD] = timeDomainHRV(ibi,xx)
%timeDomainHRV: calculates time-domain hrv of ibi interval series	

    t=ibi(:,1)-ibi(1,1);
    ibi=ibi(:,2);
    
    ibi_mean = mean(ibi);
    ibi_max = max(ibi);
    ibi_min = min(ibi);
    ibi_SDNN = std(ibi);
    ibi_RMSSD = RMSSD(ibi);
    
            
end


function output = RMSSD(ibi)
%RMSSD: root mean square of successive RR differences
   differences=abs(diff(ibi)); %successive ibi diffs 
   output=sqrt(sum(differences.^2)/length(differences));
end
