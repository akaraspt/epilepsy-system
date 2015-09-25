function [smoothRS,smoothRS_norm] = featureSmoothing(RS)
%% provide smoothed and normalized relative spectral power
%SR -- Features x time
D = size(RS);

for i = 1:D(2)
    n = max(1,i-11);
    temp = RS(:,n:i);
    temp2 = sum(temp,1);
    if isfinite(temp2(end))
        idx = find(isfinite(temp2));
        smoothRS(:,i) = mean(temp(:,idx),2);
    else
        smoothRS(:,i) = ones(D(1),1)*NaN;
    end
end

for j = 1:D(1)
    smoothRS_norm(j,:) = (smoothRS(j,:)-min(smoothRS(j,:)))/(max(smoothRS(j,:))-min(smoothRS(j,:)));
end




% smoothRS_norm = (smoothRS-min(min(smoothRS)))/(max(max(smoothRS))-min(min(smoothRS)));

