function output = phasepowerCoupling(angles,powers,sample_freq)
%% caculate phase-amplitude coupling of multiple channels


%anlges -- channel,bands,times


D_a = size(angles);
D_p = size(powers);
L = sample_freq*5;

f_indx = [];
n = 0;
phasepower = zeros(D_a(2)*D_p(2),D_a(1)*D_p(1),ceil(D_a(3)/L));
for i = 1:D_a(1)
    for j = 1:D_p(1)
        n = n+1;
        angleS = squeeze(angles(i,:,:));
        powerS = squeeze(powers(j,:,:));
        if D_a(2) == 1
            angleS = angleS';
        end
        if D_p(2) == 1
            powerS = powerS';
        end
        [phasepower(:,n,:),f_i] = phase_power(angleS,powerS,sample_freq);
        f_indx = [f_indx;repmat([i,j],size(f_i,1),1),f_i]; 
    end
end

D_r = size(phasepower);
temp = reshape(phasepower,D_r(1)*D_r(2),D_r(3));
% figure;imagesc(temp)

output.phasepower = temp;
output.f_indx = f_indx;
 