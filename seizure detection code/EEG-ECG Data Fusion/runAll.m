clear;

mkdir('mat');
addpath(genpath(pwd));
cd('./mat');

% 1. Convert the data from EDF to MAT
% and cut the data into 1-second pieces
% for the selected EEG and ECG signals
run('chbmitEdfToMat.m');

% 2. Perform time-frequency analysis
% on the selected EEG and ECG signals
run('timeFrequencyAnalysis.m');

% 3. Extract all the EEG-ECG synchronization
% features to be evaluated in the experiment
run('featureExtraction.m');

% 4. Simple feature selection
run('featureSelection.m');

% 4. Train and test without and with EEG-ECG sync
run('trainAndTest.m');