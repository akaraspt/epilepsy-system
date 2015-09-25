function  [powerpower,f_indx] = power_power(power1,power2,sample_freq)
%% caculate power-power coupling 
%angle -- bands,times

D_1 = size(power1);
D_2 = size(power2);
L = sample_freq*5;

powerpower = zeros(D_1(1)*D_2(1),ceil(D_1(2)/L));
n = 0;
for window = 1:L:L*(floor(D_1(2)/L)-1)+1
    n = n+1;
    a = 0;
    for i = 1:D_1(1)
        for j = 1:D_2(1)
                a = a + 1;
%         powerpower(i,n) = sum(temp1(i,window:window+L-1))*sum(temp2(i,window:window+L-1))/...
%             sqrt(sum(temp1(i,window:window+L-1).^2)*sum(temp2(i,window:window+L-1).^2));
                powerpower(a,n) = max(xcorr(power1(i,window:window+L-1),power2(j,window:window+L-1)));
                f_indx(a,:) = [i,j];
        end
    end
end
if mod(D_1(2),L)>0
    a = 0;
    n = n+1;
    window = window+L;
     for i = 1:D_1(1)
        for j = 1:D_2(1)
                a = a + 1;
%          powerpower(i,n) = sum(temp1(i,window:end))*sum(temp2(i,window:end))/...
%             sqrt(sum(temp1(i,window:end).^2)*sum(temp2(i,window:end).^2));
                powerpower(a,n) = max(xcorr(power1(i,window:end),power2(j,window:end)));
        end
    end
end
    