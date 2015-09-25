function [RS_power,RS_indx] = relativeSpectralPower(S_power)
%% calculate relative spectral power
%S_power -- channels x subbands x time 
%RS_indx -- channel, channel, subband, subband


D = size(S_power);

%%calculate the power rate between sub bands within a channel
n = 0;
RS_power = [];
for i = 1:D(2)-1
    for j = i+1:D(2)
        n = n+1;
        RS_power = [RS_power;S_power(:,i,:)./S_power(:,j,:)];
        temp(n,:) = [i,j];
    end
end

RS_indx = repmat(1:D(1),n,1);
RS_indx = RS_indx(:);
RS_indx = repmat(RS_indx,1,2);
RS_indx = [RS_indx,repmat(temp,D(1),1)];
clear temp

n = 0;
for i = 1:D(1)-1
    for j = 1:D(2)
        for l = i+1:D(1)
            for m = 1:D(2)
                n = n+1;
                RS_power = [RS_power;S_power(i,j,:)./S_power(l,m,:)];
                temp(n,:) = [i,l,j,m];
            end
        end
    end
end
RS_indx = [RS_indx;temp];

RS_power = squeeze(RS_power);

