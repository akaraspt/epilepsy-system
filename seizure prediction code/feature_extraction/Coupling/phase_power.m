function  [phasepower,f_indx] = phase_power(angleS,powerS,sample_freq)
%% caculate phase-amplitude coupling 
%angle -- bands,times

D_a = size(angleS);
D_p = size(powerS);
L = sample_freq*5;

phasepower = zeros(D_a(1)*D_p(1),ceil(D_a(2)/L));
n_hist_bins = ceil(1+log2(L)); % Sturges
n = 0;
for window = 1:L:L*(floor(D_a(2)/L)-1)+1
    n = n+1;
%         phasepower(i,n) = abs(mean(temp(i,window:window+L-1)));
    a = 0;
    for i = 1:D_a(1)
        for j = 1:D_p(1)
            a = a + 1;
            amp_by_phases=zeros(1,n_hist_bins);
            power = powerS(j,window:window+L-1);
            phase = angleS(i,window:window+L-1);
            phase_edges=linspace(min(phase),max(phase),n_hist_bins+1);
            for m=1:n_hist_bins-1
                indx = find(phase>phase_edges(m) & phase<phase_edges(m+1));
                if ~isempty(indx)
                    amp_by_phases(m) = mean(power(phase>phase_edges(m) & phase<phase_edges(m+1)));
                else
                    amp_by_phases(m) = 0;
                end
            end
        amp_by_phases = amp_by_phases./sum(amp_by_phases);
        ent =  -sum(amp_by_phases.*log2(amp_by_phases+eps),2);
        phasepower(a,n) = (log2(n_hist_bins)-ent)/log2(n_hist_bins);
        f_indx(a,:) = [i,j];
        end
    end
end
if mod(D_a(2),L)>0
    n = n+1;
    window = window+L;
    a = 0;
    n_hist_bins = ceil(1+log2(mod(D_a(2),L))); % Sturges
     for i = 1:D_a(1)
        for j = 1:D_p(1)
            a = a + 1;
            power = powerS(j,window:end);
            phase = angleS(i,window:end);
            phase_edges=linspace(min(phase),max(phase),n_hist_bins+1);
            for m=1:n_hist_bins-1
                indx = find(phase>phase_edges(m) & phase<phase_edges(m+1));
                if ~isempty(indx)
                    amp_by_phases(m) = mean(power(phase>phase_edges(m) & phase<phase_edges(m+1)));
                else
                    amp_by_phases(m) = 0;
                end    
            end
        amp_by_phases = amp_by_phases./sum(amp_by_phases);
        ent =  -sum(amp_by_phases.*log2(amp_by_phases+eps),2);
        phasepower(a,n) = (log2(n_hist_bins)-ent)/log2(n_hist_bins);
%         phasepower(i,n) = abs(mean(temp(i,window:end)));
        end
     end
end
    