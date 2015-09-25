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

% Record location
patient_dir = [data_dir, '/', patient_id];
selectfile = 'selectfilelist_all.txt';
seizuremetafile = 'seizurefilelist.txt';

fid = fopen([patient_dir,'/',selectfile]);
file_list = textscan(fid, '%s');
fclose(fid);

% Get metadata of all blocks
delimiter = '\t';
seizure_meta = tdfread([patient_dir, '/', seizuremetafile], delimiter);
seizure_meta.filename = cellstr(seizure_meta.filename);
seizure_meta.onset = cellstr(seizure_meta.onset);
seizure_meta.offset = cellstr(seizure_meta.offset);
seizure_meta.onset_sample = num2cell(seizure_meta.onset_sample);
seizure_meta.offset_sample = num2cell(seizure_meta.offset_sample);

% fid = fopen([patient_dir,'/',seizuremetafile]);
% seizureList = textscan(fid, '%s', 'HeaderLines', 1, 'Delimiter', '\t');
% fnameIdx = 1:5:numel(seizureList{1});
% onsetIdx = 2:5:numel(seizureList{1});
% offsetIdx = 3:5:numel(seizureList{1});
% onsetSampleIdx = 4:5:numel(seizureList{1});
% offsetSampleIdx = 5:5:numel(seizureList{1});
% seizureMeta.filename = seizureList{1}(fnameIdx);
% seizureMeta.onset = seizureList{1}(onsetIdx);
% seizureMeta.offset = seizureList{1}(offsetIdx);
% seizureMeta.onset_sample = seizureList{1}(onsetSampleIdx);
% seizureMeta.offset_sample = seizureList{1}(offsetSampleIdx);
% fclose(fid);

file_list = file_list{1};
meta_list = strrep(file_list, '.data', '.head');
save_list = strrep(file_list, '.data', '.mat');

% Convert from binary to MATLAB file including the corresponding metadata
for i=1:numel(file_list)
    disp(['Converting file ' , file_list{i} ,' ...']);
    
    data_fname = [patient_dir,'/',file_list{i}];
    meta_fname = [patient_dir,'/',meta_list{i}];
    save_fname = [patient_dir,'/',save_list{i}];
    
    %% Read metadata file
    file        = fopen(meta_fname, 'rb');
    metadata    = fgets(file);
    while ischar(metadata)
        s = strsplit(metadata,'=');
        v = matlab.lang.makeValidName(s{1});

        iswhitespace = isstrprop(s{2}(end), 'wspace');
        if iswhitespace
            s{2} = s{2}(1:end-1);
        end

        isdigit = all(isstrprop(s{2}, 'digit'));
        if isdigit || isfloat(s{2})
            eval([v ' = str2double(s{2});']);
        else
            if strcmp(v, 'conversion_factor')
                eval([v ' = str2double(s{2});']);
            else
                eval([v ' = s{2};']);
            end
        end

        metadata = fgets(file);
    end
    fclose(file);
    
    % Transform into a list of cells
    elec_names = strsplit(elec_names(2:end-1), ',')';

    %% Read in EEG data from binary file
    file        = fopen(data_fname,'rb');                                       % open file
    signals     = fread(file, [num_channels num_samples], 'int16');             % read in binary format data
    fclose(file); clear file;                                                   % close file and clear file ID
    signals     = signals.*conversion_factor;                                   % scale raw data by conversion factor

    %% Generate seizure labels
    signals_size = size(signals);
    seizure_idx = find(strcmp(file_list{i}, seizure_meta.filename));
    if numel(seizure_idx) > 0
        seizure_labels = zeros(1, signals_size(2));
        for j=1:numel(seizure_idx)
            start_seizure_idx = seizure_meta.onset_sample{seizure_idx(j)};
            end_seizure_idx = seizure_meta.offset_sample{seizure_idx(j)};
            seizure_labels(start_seizure_idx:end_seizure_idx) = 1;
        end
    else
        seizure_labels = zeros(1, signals_size(2));
    end
    
    %% Save converted data
    save(save_fname, ...
        'start_ts', 'num_samples', 'sample_freq', ...
        'conversion_factor', 'num_channels', ...
        'elec_names', 'pat_id', 'adm_id', 'rec_id', ...
        'duration_in_sec', 'sample_bytes', ...
        'signals', 'seizure_labels');

end

end
