function output = powerpowerCoupling(powers1,powers2,sample_freq)
%caculate power-power coupling of multiple channels
%anlges -- channel,bands,times


D_1 = size(powers1);
D_2 = size(powers2);
L = sample_freq*5;

f_indx = [];
n = 0;
phasepower = zeros(D_1(2)*D_2(2),D_1(1)*D_2(1),ceil(D_1(3)/L));
for i = 1:D_1(1)
    for j = 1:D_2(1)
        n = n+1;
        power1 = squeeze(powers1(i,:,:));
        power2 = squeeze(powers2(j,:,:));
        if D_1(2) == 1
            power1 = power1';
        end
        if D_2(2) == 1
            power2 = power2';
        end        
        [powerpower(:,n,:),f_i] = power_power(power1,power2,sample_freq);
        f_indx = [f_indx;repmat([i,j],size(f_i,1),1),f_i]; 
    end
end

D_r = size(powerpower);
temp = reshape(powerpower,D_r(1)*D_r(2),D_r(3));
% figure;imagesc(temp)

output.powerpower = temp;
output.f_indx = f_indx;
 