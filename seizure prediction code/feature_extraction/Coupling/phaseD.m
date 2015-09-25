function output = phaseD(angle1,angle2,Lf,Hf,Lf2,Hf2,sample_freq)
%f_indx channel, channel, band, band
%angle -- channels,bands,time

D_1 = size(angle1);
D_2 = size(angle2);

n = 0;
f_indx = [];
for i = 1:D_1(1)
    for j = 1:D_2(1)
        n = n+1;
        a1 = squeeze(angle1(i,:,:));
        a2 = squeeze(angle2(j,:,:));
        if D_1(2) == 1
            a1 = a1';
        end
        if D_2(2) ==1
            a2 = a2';
        end

        [phase(:,i,:),f_i] = phaseSynCF(a1,a2,Lf,Hf,Lf2,Hf2,sample_freq);
        f_indx = [f_indx;repmat([i,j],size(f_i,1),1),f_i]; % channel 1,channel2, subband EEG, subband IBI
    end
end
D_p = size(phase);
phaseD = reshape(phase,D_p(1)*D_p(2),D_p(3));
% figure;imagesc(phaseE)

output.phaseD = phaseD;
output.f_indx = f_indx;
