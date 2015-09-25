clear;
clc;

list_patients = { ...
    'pat_102', ...
    'pat_7302', ...
    'pat_8902', ...
    'pat_11002', ...
    'pat_16202', ...
    'pat_21902', ...
    'pat_21602', ...
    'pat_22602', ...
    'pat_23902', ...
    'pat_26102', ...
    'pat_30002', ...
    'pat_30802', ...
    'pat_32502', ...
    'pat_32702', ...
    'pat_45402', ...
    'pat_46702', ...
    'pat_55202', ...
    'pat_56402', ...
    'pat_58602', ...
    'pat_59102', ...
    'pat_75202', ...
    'pat_79502', ...
    'pat_85202', ...
    'pat_92102', ...
    'pat_93902', ...
    'pat_96002', ...
    'pat_103002', ...
    'pat_109502', ...
    'pat_111902', ...
    'pat_114902'};

for p=1:numel(list_patients)

% Parameters
patient_id = list_patients{p};
data_dir = '/Volumes/My Passport/Workspace/data/epilepsiae';
thd_flat_signal_sec = 1; % in seconds

% Record location
patient_dir = [data_dir, '/', patient_id];
selectfile = 'selectfilelist_all.txt';
seizuremetafile = 'seizurefilelist.txt';

fid = fopen([patient_dir,'/',selectfile]);
file_list = textscan(fid, '%s');
fclose(fid);

file_list = file_list{1};
data_list = strrep(file_list, '.data', '.mat');
save_list = strrep(file_list, '.data', '_flat_signal_idx.mat');

% Convert from binary to MATLAB file including the corresponding metadata
for d=1:numel(data_list)
    disp(['Get flat signal index from file ' , data_list{d} ,' ...']);
    
    data_fname = [patient_dir,'/',data_list{d}];
    save_fname = [patient_dir,'/',save_list{d}];
    
    load(data_fname);
    
    thd_flat_signal_sample = thd_flat_signal_sec * sample_freq; % Number of samples that are flat
    
    % Get a list of candidate flat signals
    df = diff(signals, 1, 2);
    sum_df = sum(df, 1);
    flat_signal_idx = find(sum_df==0);
    if numel(flat_signal_idx) > 0
        end_flat_signal_idx = find(diff(flat_signal_idx) > 1);
        
        % If there are more than one flat periods
        if numel(end_flat_signal_idx) > 0
            list_flat_signal_idx = cell(numel(end_flat_signal_idx)+1, 1);
            for l=1:numel(list_flat_signal_idx)
                if l==1
                    list_flat_signal_idx{l} = flat_signal_idx(1:end_flat_signal_idx(l));
                elseif l==numel(list_flat_signal_idx)
                    list_flat_signal_idx{l} = flat_signal_idx(end_flat_signal_idx(end)+1:end);
                else
                    list_flat_signal_idx{l} = flat_signal_idx(end_flat_signal_idx(l-1)+1:end_flat_signal_idx(l));
                end
            end
        % If there is only one flat period
        else
            list_flat_signal_idx = cell(1, 1);
            list_flat_signal_idx{1} = flat_signal_idx;
        end
    else
        list_flat_signal_idx = {};
    end

    % Remove non-flat signals and append two more samples to cover more
    % data
    flat_signal_idx = {};
    k = 1;
    for i=1:numel(list_flat_signal_idx)
        if numel(list_flat_signal_idx{i}) > thd_flat_signal_sample
            flat_signal_idx{k} = [list_flat_signal_idx{i}(1) - 1, ...
                                  list_flat_signal_idx{i}, ...
                                  list_flat_signal_idx{i}(end) + 1];
                              
            % Remove out-of-bound indices
            if flat_signal_idx{k}(1) < 1
                flat_signal_idx{k} = flat_signal_idx{k}(2:end);
            end
            if flat_signal_idx{k}(end) > size(signals, 2)
                flat_signal_idx{k} = flat_signal_idx{k}(1:end-1);
            end
            
            % Sanity check
            diff_flat_idx = setdiff(flat_signal_idx{k}, find(sum_df==0));
            diff_flat_idx = setdiff(diff_flat_idx, list_flat_signal_idx{i}(1) - 1);
            diff_flat_idx = setdiff(diff_flat_idx, list_flat_signal_idx{i}(end) + 1);
            assert(numel(diff_flat_idx) == 0);
            
            k = k + 1;
        end
    end
    
    disp(['Number of flat signal segments: ', num2str(numel(flat_signal_idx))])
    for i=1:numel(flat_signal_idx)
        disp(['[', num2str(flat_signal_idx{i}(1)), ', ', ...
              num2str(flat_signal_idx{i}(end)),']']);
    end
    
    save(save_fname, 'flat_signal_idx', ...
                     'thd_flat_signal_sec', ...
                     'thd_flat_signal_sample', ...
                     'sample_freq');
end

end
