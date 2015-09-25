
function [dibi,nibi,trend,art]=preprocessIBI(ibi)
%% preprocessing IBI signal
    
    %% correct ectopic ibi
    [nibi,art]=correctEctopic(ibi);
    
    %% Detrending
    [dibi,nibi,trend]=detrendIBI(nibi);
        
end

function [nibi,art]=correctEctopic(ibi)
    y=ibi(:,2);
    t=ibi(:,1);
    %locate ectopic
    art = locateOutliers(t,y,'percent',0.2);
    %replace ectopic
    [y t] = replaceOutliers(t,y,art,'remove');
     nibi=[t,y];
end

function [dibi,nibi,trend]=detrendIBI(ibi)
    % preallocate memory and create default trend of 0.
    y=ibi(:,2);
    t=ibi(:,1);
    nibi=[t,y];
    trend=zeros(length(y),2);
    trend(:,1)=t; %time values    
    meanIbi=mean(y);
    
    % 'polynomial' - linear detrend
    t2=(t-mean(t))./std(t);
    %fit
    polyOrder = 3; %1 -linear, 2 or 3 polynomial
    [p,S]= polyfit(t2,y,polyOrder);
    trend(:,2) = polyval(p,t2);
    dibi=[t,y-trend(:,2)]; %detrended IBI

%         %%% Note: After removing the trend, the mean value of the ibi
%         % series is near zero and some of the artifact detection methods 
%         % detect two many outlers due to a mean near zero
%         % Solution: shift detrended ibi back up to it's orignal mean
%         % by adding orignal mean to detrended ibi.                        
%          if opt.meanCorrection
             dibi(:,2) = dibi(:,2) + meanIbi; % see note above
%          end  
end%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
