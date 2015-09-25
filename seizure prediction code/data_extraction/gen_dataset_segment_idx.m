clear;
clc;

rng(1,'twister');
save_rng = rng;

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
preictal_sec = 40 * 60; % Amount of preictal data in seconds
thd_preictal_seizure_sec = 40 * 60; % Threshold to remove seizures that have short preictal periods
thd_preictal_gap_sec = 5; % Threshold to remove seizures that have non-continuous preictal periods
n_nonsz_used_each_sz = 20; % Number of non-seizure records used to get nonictal period
per_each_nonictal_file = 0.2; % Percentage of nonictal data to get from each non-seizure record
multiple_nonsz_each_sz = 1; % Multiple of nonictal data for each ictal (can resampling to balanced the number of classes later)
n_extended_blocks_test = 2; % Number of blocks to be extended for cv and testing

% Used to check that all training data have the same sampling rate
sampling_rate = -1;

% Record location
patient_dir = [data_dir, '/', patient_id];
blockmetafile = 'block_metadata.txt';
seizuremetafile = 'seizurefilelist.txt';

% Get metadata of all blocks
delimiter = '\t';
block_meta = tdfread([patient_dir, '/', blockmetafile], delimiter);
block_meta.begin_dt = cellstr(block_meta.begin_dt);
block_meta.end_dt = cellstr(block_meta.end_dt);
block_meta.filename = cellstr(block_meta.filename);
block_meta.format = cellstr(block_meta.format);
block_meta.checksum = cellstr(block_meta.checksum);

% For format consistent
block_meta.begin_dt = strrep(block_meta.begin_dt, '/', '-');
block_meta.end_dt = strrep(block_meta.end_dt, '/', '-');

% Get metadata of all seizures
delimiter = '\t';
seizure_meta = tdfread([patient_dir, '/', seizuremetafile], delimiter);
seizure_meta.filename = cellstr(seizure_meta.filename);
seizure_meta.onset = cellstr(seizure_meta.onset);
seizure_meta.offset = cellstr(seizure_meta.offset);

%% Select candidate seizures
%   - Ignore seizures that have preictal periods less than the specified
%   threshold
%   - Ignore seizures that do not have continuous preictal periods (i.e.
%   have large gaps in the preictal periods)

disp('==============================');
disp('All seizure files');
disp('==============================');
disp(seizure_meta.filename);

disp('==============================');
disp('Remove seizures with short preictal ...');
disp('==============================');

% Compute the inter-ictal period 
d1 = datevec(seizure_meta.offset(1:end-1));
d2 = datevec(seizure_meta.onset(2:end));
elapsed_sec = etime(d2,d1);

% Seizures that have preictal period less than the specified threshold will be ignored
long_preictal_sz = elapsed_sec > thd_preictal_seizure_sec;
long_preictal_sz = [true; long_preictal_sz]; % add the first seizure
select_sz_idx = find(long_preictal_sz);

% Check the first seizure in the case that there is no enough previous records (i.e., it
% reaches the first record of the patient) to get preictal data
first_select_sz_idx = select_sz_idx(1);
first_sz_onset = datevec(seizure_meta.onset(first_select_sz_idx));
first_block_begin_dt = datevec(block_meta.begin_dt(1));
first_sz_longest_preictal_sec = etime(first_sz_onset, first_block_begin_dt);
if first_sz_longest_preictal_sec < thd_preictal_seizure_sec
    select_sz_idx = select_sz_idx(2:end);
end

disp('Remove: ');
disp(setdiff(1:numel(seizure_meta.filename), select_sz_idx)');
disp('Remain: ');
disp(select_sz_idx);

% disp('==============================');
% disp('Remove seizures with large gap before seizure records');
% disp('==============================');
% 
% % Seizures that have large gaps in the preictal periods
% non_con_preictal_block = block_meta.gap >= thd_preictal_gap_sec;
% non_con_preictal_block_file = block_meta.filename(non_con_preictal_block);
% select_sz_file = setdiff(long_preictal_sz_file, non_con_preictal_block_file);
% 
% disp('Remove: ');
% disp(intersect(long_preictal_sz_file, non_con_preictal_block_file));
% disp('Remain: ');
% disp(select_sz_file);

disp('==============================');
disp('Remove seizures that have non-continuous preictal');
disp('==============================');

non_con_preictal_sz_idx = [];
for i=1:numel(select_sz_idx)
    sz_idx = select_sz_idx(i);
    
    % Get an index of the seizure's onset
    sz_onset = seizure_meta.onset_sample(sz_idx);
    
    % Get indices of the preictal in the seizure record
    preictal_in_sz_idx = 1:sz_onset-1;
    
    % Compute duration of the preictal period in the seizure record
    flat_seg_fname = strrep(seizure_meta.filename{sz_idx}, '.data', '_flat_signal_segment_idx.mat');
    load([patient_dir, '/', flat_seg_fname], 'sample_freq', 'segment_sec');
    if sampling_rate == -1
        sampling_rate = sample_freq;
    else
        assert(sampling_rate == sample_freq, ...
               'Sampling rate is not the same for all blocks in the training data.');
    end
    n_preictal_segments = preictal_sec / segment_sec;
    preictal_duration = numel(preictal_in_sz_idx) / sample_freq;
    n_segments = floor(preictal_duration / segment_sec);
    
    % Check the gap before the seizure record if the amount of preictal is
    % not sufficient
    n_remains = n_preictal_segments - n_segments;
    block_sz_idx = find(strcmp(seizure_meta.filename(sz_idx), block_meta.filename));
    if (n_remains > 0) && (block_meta.gap(block_sz_idx) >= thd_preictal_gap_sec)
        non_con_preictal_sz_idx = [non_con_preictal_sz_idx; sz_idx];
    else
        % Check the gap of the record before seizure
        n_block_before_sz = 1;
        while(n_remains > 0)
            block_nonsz_idx = find(strcmp(seizure_meta.filename(sz_idx), block_meta.filename)) - n_block_before_sz;
            block_nonsz_n_samples = block_meta.samples(block_nonsz_idx);
            block_nonsz_durations = block_nonsz_n_samples / sample_freq;
            block_nonsz_n_segments = ceil(block_nonsz_durations / segment_sec);
            n_remains = n_remains - block_nonsz_n_segments;

            % If there is a large gap in the preictal, disgard this seizure
            if (n_remains > 0) && (block_meta.gap(block_nonsz_idx) >= thd_preictal_gap_sec)
                non_con_preictal_sz_idx = [non_con_preictal_sz_idx; sz_idx];
                break;
            end

            n_block_before_sz = n_block_before_sz + 1;
        end
    end
end

% Further remove the remaining seizures that have non-continuous preictal
select_sz_idx = setdiff(select_sz_idx, non_con_preictal_sz_idx);

disp('Remove: ');
disp(non_con_preictal_sz_idx);

disp('Remain: ');
disp(select_sz_idx);


%% Get preictal periods

all_sz_idx = 1:numel(seizure_meta.filename);

disp('==============================');
disp('Get preictal periods ...');
disp('==============================');
preictals = cell(numel(all_sz_idx), 1);
for i=1:numel(all_sz_idx)
    sz_idx = all_sz_idx(i);
    
    % Get an index of the seizure's onset
    sz_onset = seizure_meta.onset_sample(sz_idx);
    
    % Get indices of the preictal in the seizure record
    preictal_in_sz_idx = 1:sz_onset-1;
    
    % Compute duration of the preictal period in the seizure record
    flat_seg_fname = strrep(seizure_meta.filename{sz_idx}, '.data', '_flat_signal_segment_idx.mat');
    load([patient_dir, '/', flat_seg_fname], 'sample_freq', 'segment_sec');
    assert(sampling_rate == sample_freq, ...
           'Sampling rate is not the same for all blocks in the training data.');
    n_preictal_segments = preictal_sec / segment_sec;
    preictal_duration = numel(preictal_in_sz_idx) / sample_freq;
    
    % Get preictal periods
    end_preictal_seg_idx = floor(preictal_duration / segment_sec);
    start_preictal_seg_idx = end_preictal_seg_idx - n_preictal_segments + 1;
    if start_preictal_seg_idx <= 0
        start_preictal_seg_idx = 1;
    end
    preictal_idx = start_preictal_seg_idx:end_preictal_seg_idx;
    
    p_idx = 1;
    preictals{i}.filename{p_idx} = seizure_meta.filename{sz_idx};
    preictals{i}.raw_idx{p_idx} = preictal_idx;
    p_idx = p_idx + 1;

    block_sz_idx = find(strcmp(seizure_meta.filename(sz_idx), block_meta.filename));
    disp(['Get preictal for seizure ', num2str(sz_idx), ' from ', seizure_meta.filename{sz_idx}, ...
          ' ', num2str(numel(preictal_idx)), '(', num2str(start_preictal_seg_idx) ', ', ...
          num2str(end_preictal_seg_idx), ') gap before is ', ...
          num2str(block_meta.gap(block_sz_idx))]);
    
    % Get indices of the preictal in the record before seizure (if
    % necessary)
    n_remains = n_preictal_segments - numel(preictal_idx);
    n_block_before_sz = 1;
    while(n_remains > 0)
        % Get preictal data from the non-seizure record
        block_nonsz_idx = find(strcmp(seizure_meta.filename{sz_idx}, block_meta.filename)) - n_block_before_sz;
        
        % If there is no previous record (i.e., it reaches the first record of the patient)
        if (n_remains > 0) && (block_nonsz_idx < 1)
            break;
        end
        
        block_nonsz_n_samples = block_meta.samples(block_nonsz_idx);
        block_nonsz_durations = block_nonsz_n_samples / sample_freq;
        block_nonsz_end_preictal_seg_idx = ceil(block_nonsz_durations / segment_sec);
        block_nonsz_start_preictal_seg_idx = block_nonsz_end_preictal_seg_idx - n_remains + 1;
        if block_nonsz_start_preictal_seg_idx <= 0
            block_nonsz_start_preictal_seg_idx = 1;
        end
        block_nonsz_preictal_idx = block_nonsz_start_preictal_seg_idx:block_nonsz_end_preictal_seg_idx;
        preictals{i}.filename{p_idx} = block_meta.filename{block_nonsz_idx};
        preictals{i}.raw_idx{p_idx} = block_nonsz_preictal_idx;
        p_idx = p_idx + 1;
        
        disp(['Get preictal for seizure ', num2str(sz_idx), ' from ', block_meta.filename{block_nonsz_idx}, ...
              ' ', num2str(numel(block_nonsz_preictal_idx)), '(', num2str(block_nonsz_start_preictal_seg_idx) ', ', ...
              num2str(block_nonsz_end_preictal_seg_idx), ') gap before is ', ...
              num2str(block_meta.gap(block_nonsz_idx))]);
        
        % Update parameters
        n_remains = n_remains - numel(block_nonsz_preictal_idx);
        n_block_before_sz = n_block_before_sz + 1;
    end
end

clearvars -except preictals block_meta seizure_meta ...
                  segment_sec preictal_sec n_preictal_segments ...
                  thd_preictal_seizure_sec thd_preictal_gap_sec sampling_rate ...
                  n_nonsz_used_each_sz per_each_nonictal_file patient_dir ...
                  select_sz_idx multiple_nonsz_each_sz list_patients n_extended_blocks_test


%% Remove abnormal signal during the preictal

disp(' ');
disp('==============================');
disp('Remove abnormal signal in the preictal ...');
disp('==============================');

for i=1:numel(preictals)
    for j=1:numel(preictals{i}.filename)
        preictal_file = preictals{i}.filename{j};
        preictal_idx = preictals{i}.raw_idx{j};
        
        % Get flat segment indices
        flat_seg_fname = strrep(preictal_file, '.data', '_flat_signal_segment_idx.mat');
        load([patient_dir, '/', flat_seg_fname]);
        assert(sampling_rate == sample_freq, ...
               'Sampling rate is not the same for all blocks in the training data.');
        
        % Remove flat segments from preictal
        clean_preictal_idx = preictal_idx;
        for k=1:numel(flat_signal_segment_idx)
            clean_preictal_idx = setdiff(clean_preictal_idx, flat_signal_segment_idx{k});
        end
        
        disp(['Remove flat signal segments from preictal in ', preictal_file]);
        disp(setdiff(preictal_idx, clean_preictal_idx));
        
        % Save clean preictal indices
        preictals{i}.idx{j} = clean_preictal_idx;
    end
end

clearvars -except nbins preictals block_meta seizure_meta ...
                  segment_sec preictal_sec n_preictal_segments ...
                  thd_preictal_seizure_sec thd_preictal_gap_sec sampling_rate ...
                  n_nonsz_used_each_sz per_each_nonictal_file patient_dir ...
                  select_sz_idx multiple_nonsz_each_sz list_patients n_extended_blocks_test
              
all_preictals = preictals;
preictals = all_preictals(select_sz_idx);

disp(' ');

% Remove overlapped all_preictal data
for i=1:numel(all_preictals)
    for j=1:numel(all_preictals{i}.filename)
        for k=i+1:numel(all_preictals)
            for l=1:numel(all_preictals{k}.filename)
                if strcmp(all_preictals{i}.filename{j}, all_preictals{k}.filename{l})
                    nonoverlap_idx = setdiff(all_preictals{k}.idx{l}, all_preictals{i}.idx{j});
                    if ~isequal(nonoverlap_idx, all_preictals{k}.idx{l})
                        disp(['Remove all_preictals ', all_preictals{i}.filename{j}, ' from overlapped all_preictals ', all_preictals{k}.filename{l}]);
                        all_preictals{k}.idx{l} = nonoverlap_idx;
                    end
                end
            end
        end
    end
end

% Remove overlapped preictal data from all_preictals
remove_sz_idx = setdiff(1:numel(all_preictals), select_sz_idx);
for i=1:numel(remove_sz_idx)
    i1 = remove_sz_idx(i);
    for j=1:numel(all_preictals{i1}.filename)
        for k=1:numel(preictals)
            for l=1:numel(preictals{k}.filename)
                if strcmp(all_preictals{i1}.filename{j}, preictals{k}.filename{l})
                    nonoverlap_idx = setdiff(all_preictals{i1}.idx{j}, preictals{k}.idx{l});
                    all_preictals{i1}.idx{j} = nonoverlap_idx;
                end
            end
        end
    end
end


%% Randomly select non-ictal periods with balanced classes

disp(' ');
disp('==============================');
disp('Randomly select non-ictal periods with balanced classes ...');
disp('==============================');

preictal_filename = {};
for i=1:numel(preictals)
    preictal_filename = [preictal_filename, preictals{i}.filename];
    
    % Concatenate blocks before and after each preictal as they will be 
    % used in the testing
    block_sz_idx = find(strcmp(preictals{i}.filename{1}, block_meta.filename));
    list_concate_block_idx = block_sz_idx-n_extended_blocks_test:block_sz_idx+n_extended_blocks_test;
    for k=1:numel(list_concate_block_idx)
        if list_concate_block_idx(k) > numel(block_meta.filename)
            continue
        end
        if list_concate_block_idx(k) < 1
            continue
        end
        preictal_filename = [preictal_filename, block_meta.filename{list_concate_block_idx(k)}];
    end
end
% Add (if any) seizures that have been removed due to short and
% non-continuous preictals
for i=1:numel(seizure_meta.filename)
    preictal_filename = [preictal_filename, seizure_meta.filename{i}];
    
    % Concatenate blocks before each preictal as they might contain some 
    % preictal data
    block_sz_idx = find(strcmp(seizure_meta.filename{i}, block_meta.filename));
    block_sz_idx = block_sz_idx - 1;
    if block_sz_idx > 0
        preictal_filename = [preictal_filename, block_meta.filename{block_sz_idx}];
    end
end
preictal_filename = unique(preictal_filename);

nonictal_files = setdiff(block_meta.filename, preictal_filename);
nonictal_perm_idx = randperm(numel(nonictal_files));
nonictal_block_idx = 1;
nonictals = cell(numel(preictals), 1);
for i=1:numel(preictals)
    % Count the number of indicies for the preictal period of each seizure
    n_segments = 0;
    for j=1:numel(preictals{i}.idx)
        n_segments = n_segments + numel(preictals{i}.idx{j});
    end
    preictals{i}.n_segments = n_segments;
    disp(['Preictal segment for seizure from ', preictals{i}.filename{1}, ...
          ': ', num2str(n_segments)]);
      
    % Random blocks with replacement
    start_perm_idx = ((nonictal_block_idx-1)*n_nonsz_used_each_sz) + 1;
    end_perm_idx = nonictal_block_idx*n_nonsz_used_each_sz;
    if end_perm_idx > numel(nonictal_files)
        nonictal_perm_idx = randperm(numel(nonictal_files));
        nonictal_block_idx = 1;
        start_perm_idx = ((nonictal_block_idx-1)*n_nonsz_used_each_sz) + 1;
        end_perm_idx = nonictal_block_idx*n_nonsz_used_each_sz;
    end
    nonictal_block_idx = nonictal_block_idx + 1;
    nonictal_filename = nonictal_files(nonictal_perm_idx(start_perm_idx:end_perm_idx));
    
    n_all_nonictal = 0;
    for j=1:numel(nonictal_filename)
        
        % Get flat segment indices
        flat_seg_fname = strrep(nonictal_filename{j}, '.data', '_flat_signal_segment_idx.mat');
        load([patient_dir, '/', flat_seg_fname]);
        assert(sampling_rate == sample_freq, ...
               'Sampling rate is not the same for all blocks in the training data.');
       
        % Get random indices
        block_idx = find(strcmp(nonictal_filename{j}, block_meta.filename));
        n_nonictal_segments = ceil(block_meta.samples(block_idx) / (sample_freq * segment_sec));
        normal_segment_idx = 1:n_nonictal_segments;
        for k=1:numel(flat_signal_segment_idx)
            normal_segment_idx = setdiff(normal_segment_idx, flat_signal_segment_idx{k});
        end
        normal_segment_perm_idx = randperm(numel(normal_segment_idx));
        n_nonictal = ceil(numel(normal_segment_idx) * per_each_nonictal_file);
        
        if (n_all_nonictal + n_nonictal) > (n_segments * multiple_nonsz_each_sz)
            n_nonictal = (n_segments * multiple_nonsz_each_sz) - n_all_nonictal;
        end
        
        if n_nonictal == 0 && n_all_nonictal == (n_segments * multiple_nonsz_each_sz)
            break;
        end
        
        nonictals{i}.filename{j} = nonictal_filename{j};
        nonictals{i}.idx{j} = normal_segment_idx(normal_segment_perm_idx(1:n_nonictal));
        
        disp(['Nonictal segment for this seizure from ', nonictals{i}.filename{j}, ...
              ': ', num2str(numel(nonictals{i}.idx{j}))]);
        
        n_all_nonictal = n_all_nonictal + n_nonictal;
        
        if n_all_nonictal == n_segments
            break;
        end
    end
    
    nonictals{i}.n_segments = n_all_nonictal;
end

clearvars -except preictals nonictals block_meta seizure_meta ...
                  segment_sec preictal_sec n_preictal_segments ...
                  thd_preictal_seizure_sec thd_preictal_gap_sec sampling_rate ...
                  n_nonsz_used_each_sz per_each_nonictal_file patient_dir ...
                  select_sz_idx multiple_nonsz_each_sz list_patients nonictal_files ...
                  n_extended_blocks_test all_preictals


%% Get ictal periods

all_sz_idx = 1:numel(all_preictals);

disp(' ');
disp('==============================');
disp('Get ictal periods ...');
disp('==============================');
ictals = cell(numel(all_sz_idx), 1);
for i=1:numel(all_sz_idx)
    sz_idx = all_sz_idx(i);
    
    % Get flat segment indices
    flat_seg_fname = strrep(seizure_meta.filename{sz_idx}, '.data', '_flat_signal_segment_idx.mat');
    load([patient_dir, '/', flat_seg_fname]);
    assert(sampling_rate == sample_freq, ...
           'Sampling rate is not the same for all blocks in the training data.');
    
    % Get an index of the seizure's indices
    sz_onset = seizure_meta.onset_sample(sz_idx);
    sz_offset = seizure_meta.offset_sample(sz_idx);
    
    % Get segment indices of the ictal
    start_ictal_seg_idx = ceil(sz_onset / (sample_freq * segment_sec));
    end_ictal_seg_idx = ceil(sz_offset / (sample_freq * segment_sec));
    
    % Get number of segments
    block_idx = find(strcmp(seizure_meta.filename{sz_idx}, block_meta.filename));
    n_segments = ceil(block_meta.samples(block_idx) / (sample_freq * segment_sec));
    
    % Ignore the ictal segments that are beyond this block
    if end_ictal_seg_idx > n_segments
        end_ictal_seg_idx = n_segments;
    end
        
    % Get ictal periods
    ictal_idx = start_ictal_seg_idx:end_ictal_seg_idx;
    
    % Remove flat segments from ictal
    clean_ictal_idx = ictal_idx;
    for k=1:numel(flat_signal_segment_idx)
        clean_ictal_idx = setdiff(clean_ictal_idx, flat_signal_segment_idx{k});
    end
    
    if numel(clean_ictal_idx) == 0
        disp(['Get ictal from seizure ', num2str(sz_idx), ' from ', seizure_meta.filename{sz_idx}, ...
              ' ', num2str(numel(clean_ictal_idx)), '()']);
    else
        disp(['Get ictal from seizure ', num2str(sz_idx), ' from ', seizure_meta.filename{sz_idx}, ...
              ' ', num2str(numel(clean_ictal_idx)), '(', num2str(clean_ictal_idx(1)) ', ', ...
              num2str(clean_ictal_idx(end)), ')']);
    end
    
    if numel(clean_ictal_idx) ~= numel(ictal_idx)
        disp(['Remove flat signal segments from ictal in ', seizure_meta.filename{sz_idx}]);
        disp(setdiff(ictal_idx, clean_ictal_idx));
    end
    
    p_idx = 1;
    ictals{i}.filename{p_idx} = seizure_meta.filename{sz_idx};
    ictals{i}.raw_idx{p_idx} = ictal_idx;
    ictals{i}.idx{p_idx} = clean_ictal_idx;
    p_idx = p_idx + 1;
end

% Count the number of indicies for the ictal period of each seizure
for i=1:numel(ictals)
    n_segments = 0;
    for j=1:numel(ictals{i}.idx)
        n_segments = n_segments + numel(ictals{i}.idx{j});
    end
    ictals{i}.n_segments = n_segments;
end

all_ictals = ictals;
ictals = all_ictals(select_sz_idx);

disp(' ');

% Remove ictal indices that might appear in preictal due to two seizures
% occur very close to each other
for i=1:numel(all_ictals)
    for j=1:numel(all_ictals{i}.filename)
        % All preictal
        for k=1:numel(all_preictals)
            for l=1:numel(all_preictals{k}.filename)
                if strcmp(all_ictals{i}.filename{j}, all_preictals{k}.filename{l})
                    nonoverlap_idx = setdiff(all_preictals{k}.idx{l}, all_ictals{i}.idx{j});
                    if ~isequal(nonoverlap_idx, all_preictals{k}.idx{l})
                        disp(['Remove all_ictals ', all_ictals{i}.filename{j}, ' from overlapped all_preictals ', all_preictals{k}.filename{l}]);
                        all_preictals{k}.idx{l} = nonoverlap_idx;
                    end
                end
            end
        end
        
        % Preictal
        for k=1:numel(preictals)
            for l=1:numel(preictals{k}.filename)
                if strcmp(all_ictals{i}.filename{j}, preictals{k}.filename{l})
                    nonoverlap_idx = setdiff(preictals{k}.idx{l}, all_ictals{i}.idx{j});
                    if ~isequal(nonoverlap_idx, preictals{k}.idx{l})
                        disp(['Remove all_ictals ', all_ictals{i}.filename{j}, ' from overlapped preictals ', all_preictals{k}.filename{l}]);
                        preictals{k}.idx{l} = nonoverlap_idx;
                    end
                end
            end
        end
    end
end

% Sanity check
for i=1:numel(ictals)
    intersect_pre_ic = intersect(ictals{i}.raw_idx{1}, preictals{i}.idx{1});
    assert(numel(intersect_pre_ic) == 0);
    assert(strcmp(preictals{i}.filename{1}, ictals{i}.filename{1}) == 1);
    assert(preictals{i}.raw_idx{1}(end) < ictals{i}.raw_idx{1}(1));
    assert(preictals{i}.raw_idx{1}(end) == ictals{i}.raw_idx{1}(1) - 1);
    assert(preictals{i}.idx{1}(end) < ictals{i}.idx{1}(1));
end

clearvars -except preictals nonictals ictals ...
                  block_meta seizure_meta ...
                  segment_sec preictal_sec n_preictal_segments ...
                  thd_preictal_seizure_sec thd_preictal_gap_sec sampling_rate ...
                  n_nonsz_used_each_sz per_each_nonictal_file patient_dir ...
                  select_sz_idx multiple_nonsz_each_sz list_patients nonictal_files ...
                  all_ictals n_extended_blocks_test all_preictals


%% Get all nonictal periods

disp(' ');
disp('==============================');
disp('Generate all non-ictal periods for each fold ...');
disp('==============================');

nonictals_all = cell(numel(preictals), 1);
for j=1:numel(nonictals_all)
    nonictals_all{j}.filename = cell(1, numel(block_meta.filename));
    nonictals_all{j}.idx = cell(1, numel(block_meta.filename));
    for i=1:numel(block_meta.filename)

        % Get flat segment indices
        flat_seg_fname = strrep(block_meta.filename{i}, '.data', '_flat_signal_segment_idx.mat');
        load([patient_dir, '/', flat_seg_fname]);
        assert(sampling_rate == sample_freq, ...
               'Sampling rate is not the same for all blocks in the training data.');

        % Get all indices
        n_nonictal_segments = ceil(block_meta.samples(i) / (sample_freq * segment_sec));
        normal_segment_idx = 1:n_nonictal_segments;
        for k=1:numel(flat_signal_segment_idx)
            normal_segment_idx = setdiff(normal_segment_idx, flat_signal_segment_idx{k});
        end

        nonictals_all{j}.filename{i} = block_meta.filename{i};
        nonictals_all{j}.idx{i} = normal_segment_idx;
    end
    
    % Remove preictal data
    for i=1:numel(all_preictals)
        for k=1:numel(all_preictals{i}.filename)
            non_all_idx = find(strcmp(all_preictals{i}.filename{k}, nonictals_all{j}.filename));
            new_idx = setdiff(nonictals_all{j}.idx{non_all_idx}, all_preictals{i}.idx{k});
            nonictals_all{j}.idx{non_all_idx} = new_idx;
        end
    end
    
    % Remove ictal data
    for i=1:numel(all_ictals)
        for k=1:numel(all_ictals{i}.filename)
            non_all_idx = find(strcmp(all_ictals{i}.filename{k}, nonictals_all{j}.filename));
            new_idx = setdiff(nonictals_all{j}.idx{non_all_idx}, all_ictals{i}.idx{k});
            nonictals_all{j}.idx{non_all_idx} = new_idx;
        end
    end
    
    % Remove extended blocks for the leave-out-seizure
    rm_non_idx = [];
    non_all_idx = find(strcmp(preictals{j}.filename{1}, nonictals_all{j}.filename));
    list_concate_block_idx = non_all_idx-n_extended_blocks_test:non_all_idx+n_extended_blocks_test;
    for k=1:numel(list_concate_block_idx)
        if list_concate_block_idx(k) > numel(nonictals_all{j}.filename)
            continue
        end
        if list_concate_block_idx(k) < 1
            continue
        end
        rm_non_idx = [rm_non_idx, list_concate_block_idx(k)];
        disp(['Remove segment from ', nonictals_all{j}.filename{list_concate_block_idx(k)}, ...
              ' for fold ', num2str(j)])
    end
    keep_non_idx = 1:numel(nonictals_all{j}.filename);
    keep_non_idx = setdiff(keep_non_idx, rm_non_idx);
    nonictals_all{j}.filename = nonictals_all{j}.filename(keep_non_idx);
    nonictals_all{j}.idx = nonictals_all{j}.idx(keep_non_idx);
    
end


%% Summary

disp(' ');
disp('==============================');
disp('Summary');
disp('==============================');
for i=1:numel(ictals)
    disp(['Seizure ', num2str(select_sz_idx(i)), ': ictal(', num2str(ictals{i}.n_segments), '), preictal(', num2str(preictals{i}.n_segments), '), nonictal(', num2str(nonictals{i}.n_segments), ')'])
end

clearvars -except preictals nonictals ictals ...
                  block_meta seizure_meta ...
                  segment_sec preictal_sec n_preictal_segments ...
                  thd_preictal_seizure_sec thd_preictal_gap_sec sampling_rate ...
                  n_nonsz_used_each_sz per_each_nonictal_file patient_dir ...
                  select_sz_idx multiple_nonsz_each_sz list_patients nonictal_files ...
                  all_ictals n_extended_blocks_test all_preictals nonictals_all

% save([patient_dir, '/', 'trainset_', num2str(preictal_sec), '.mat'], ...
%      'preictals', 'nonictals', 'ictals', ...
%      'block_meta', 'seizure_meta', ...
%      'segment_sec', 'preictal_sec', 'n_preictal_segments', ...
%      'thd_preictal_seizure_sec', 'thd_preictal_gap_sec', 'sampling_rate', ...
%      'n_nonsz_used_each_sz', 'per_each_nonictal_file', 'patient_dir', ...
%      'select_sz_idx', 'multiple_nonsz_each_sz', 'nonictal_files', ...
%      'nonictals_all', 'n_extended_blocks_test', 'all_preictals', 'all_ictals');

end









