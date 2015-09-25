# Seizure Prediction System
Our seizure prediction system consists of three main components:
 1. [Data extraction](#data_extraction)
 2. [Feature extraction](#feature_extraction)
 3. [Seizure prediction](#seizure_prediction)
 
Note:
 - We provide this source code to illustrate the algorithms and their application
 - Depending on the weight parameters that are randomly initialized at the beginning, the results might be varied
 
### Configuration files
You have to change two parameters in `utils/common_params.py` to specify directories of the data and the system:
 - `DATA_DIR`: directory of the clinical dataset
 - `SYSTEM_DIR`: directory of the system
 
### Requirements
This component is implemented with Python 2.7 and MATLAB 2014.

The required packages for this system are as follows:
```
[Python package name]==[version]
backports.ssl-match-hostname==3.4.0.2
Cython==0.22
decorator==3.4.0
ipython==3.0.0
Jinja2==2.7.3
jsonschema==2.4.0
MarkupSafe==0.23
matplotlib==1.4.3
mistune==0.5.1
networkx==1.9.1
nose==1.3.4
numpy==1.9.2
pandas==0.15.2
Pillow==2.7.0
ptyprocess==0.4
pydot==1.0.28
Pygments==2.0.2
-e git://github.com/lisa-lab/pylearn2.git@c7ebe72cab2e1abc6d7bdf931728009c52478791#egg=pylearn2-master
pyparsing==1.5.6
python-dateutil==2.4.1
pytz==2014.9
PyYAML==3.11
pyzmq==14.5.0
scikit-image==0.11.2
scikit-learn==0.15.2
scipy==0.15.1
six==1.9.0
terminado==0.5
Theano==0.7.0rc2
tornado==4.1
wheel==0.24.0
```
Some of these packages depend on CUDA 7.0 and cuDNN 6.5 from NVIDIA.
Please make sure that both of them are installed before using this sytem.
 
## <a name="data_extraction"></a>Data Extraction
This component is used to convert raw data from Epilepsiae database into .MAT files, and generate indices for training and test sets.

### How to run this component
1. Download patient data from [Epilepsiae dataset](http://www.epilepsiae.eu/)
2. Save patient data according to [the provided structure](#epilepsiae_structure)
3. Copy metadata files in `epilepsiae` directory, and place them according to [the provided structure](#epilepsiae_structure)
4. Run `data_extraction/convert_to_mat.m` to convert from raw data into .MAT files
 - Specify a list of patients in `list_patients`
 - Change `data_dir` to the directory of the downloaded Epilepsiae dataset
5. Run `data_extraction/gen_flat_signal.m` to generate indices of flat (or abnormal) signal
 - Specify a list of patients in `list_patients`
 - Change `data_dir` to the directory of the downloaded Epilepsiae dataset
6. Run `data_extraction/gen_flat_signal_segment_idx.m` to generate segment (or sample) indices of flat signal
 - Specify a list of patients in `list_patients`
 - Change `data_dir` to the directory of the downloaded Epilepsiae dataset
 - Specify a length of each segment from which to extract features in `segment_sec`
7. Run `data_extraction/gen_dataset_segment_idx.m` to generate indicies for training and test sets
 - Specify a list of patients in `list_patients`
 - Change `data_dir` to the directory of the downloaded Epilepsiae dataset
 - Specify parameters used to generate training and test sets: `preictal_sec`, `thd_preictal_seizure_sec`, `thd_preictal_gap_sec`, `n_nonsz_used_each_sz`, `per_each_nonictal_file`, `multiple_nonsz_each_sz`, and `n_extended_blocks_test`

Note: 
 - The output of the data extraction should be stored according to [the provided structure](#epilepsiae_structure)
 - The default parameters are the ones that used in the experiment

### <a name="epilepsiae_structure"></a>Structure of Epilepsiae Dataset
```
epilepsiae
├── pat_102
│   ├── adm_1102
│   │   └── rec_100102
│   │       ├── 100102_0050.data        (from Epilepsiae dataset)
│   │       ├── 100102_0050.head        (from Epilepsiae dataset)
│   │       ├── 100102_0050.mat
│   │       ├── 100102_0050Features.mat
│   │       ├── 100102_0050_flat_signal_idx.mat
│   │       ├── 100102_0050_flat_signal_segment_idx.mat
│   │       ├── 100102_0051.data        (from Epilepsiae dataset)
│   │       ├── 100102_0051.head        (from Epilepsiae dataset)
│   │       ├── 100102_0051.mat
│   │       ├── 100102_0051Features.mat
│   │       ├── 100102_0051_flat_signal_idx.mat
│   │       ├── 100102_0051_flat_signal_segment_idx.mat
│   │       ...
│   ├── block_metadata.txt              (metadata file)
│   ├── checksums_data.txt              (from Epilepsiae dataset)
│   ├── checksums_head.txt              (from Epilepsiae dataset)
│   ├── checksums_mri.txt               (from Epilepsiae dataset)
│   ├── seizurefilelist.txt             (metadata file)
│   ├── selectfilelist_all.txt          (metadata file)
│   ├── trainset_1200.mat
│   ├── trainset_1800.mat
│   ├── trainset_2400.mat
│   └── trainset_600.mat
├── pat_16202
│   ├── adm_162102
│   │   ├── rec_16200102
│   │   │   ├── 16200102_0000.data      (from Epilepsiae dataset)
│   │   │   ├── 16200102_0000.head      (from Epilepsiae dataset)
│   │   │   ├── 16200102_0000.mat
│   │   │   ├── 16200102_0000Features.mat
│   │   │   ├── 16200102_0000_flat_signal_idx.mat
│   │   │   ├── 16200102_0000_flat_signal_segment_idx.mat
│   │   │   ...
│   │   ├── rec_16201102
│   │   │   ├── 16201102_0000.data      (from Epilepsiae dataset)
│   │   │   ├── 16201102_0000.head      (from Epilepsiae dataset)
│   │   │   ├── 16201102_0000.mat
│   │   │   ├── 16201102_0000Features.mat
│   │   │   ├── 16201102_0000_flat_signal_idx.mat
│   │   │   ├── 16201102_0000_flat_signal_segment_idx.mat
│   │   │   ...
│   ├── block_metadata.txt              (metadata file)
│   ├── checksums_data.txt              (from Epilepsiae dataset)
│   ├── checksums_head.txt              (from Epilepsiae dataset)
│   ├── checksums_mri.txt               (from Epilepsiae dataset)
│   ├── seizurefilelist.txt             (metadata file)
│   ├── selectfilelist_all.txt          (metadata file)
│   ├── trainset_1200.mat
│   ├── trainset_1800.mat
│   ├── trainset_2400.mat
│   └── trainset_600.mat
...
```

## <a name="feature_extraction"></a>Feature Extraction
This component is used to extract features from patient data stored in .MAT file. It assumes that you have already done [Data Extraction](#data_extraction).

### How to run this component
1. Run `featureExtracionMain.m`
 - Change from `pwd` MATLAB command for identifying current folder to be the directory of the patient data
 - Specify which patient to extract features from in `PAT`
 - Specify indicies of blocks to extract features from in `N`
   - E.g., for patient 1, `N = [50:222]`, for patient 325, `N = [0:200]`
 - All extracted features will be in a cell, called `output`, and the results will be saved in folder `pat_[PATIENT_ID]Feature` (e.g. `pat_32702Feature` for patient 327)
2. Copy or move the output files to the directory of the downloaded Epilepsiae dataset according to [the provided structure](#epilepsiae_structure)

### Description of Important Files
- `featureExtraction.m` - contains a function to extract all features from a recording with defined input folder and output folder
 - All functions for extraction features of relative spectral power, coupling, heart rate variability analysis are in `RelatvieSpectralPower`, `Coupling`, and `HeartRateVariability` directories respectively.
 - All other useful functions are saved in folder `Others`
- `Clinical_data_seizures.xlsx` - contains a list of the origins of each seizures of patients, which are used for channel selection
- `10_1020systemCoordinates.mat` - contains a distance map of electrodes contained by 10/20 and 10/10 system, used for channel selection 
 
### Description of Output Features
The following are parameters in the `output` variable.
```
HRV - Features of Heart Rate Variability Analysis
    ibi - 1st column is the time index of R wave, and the 2nd column is the heart beat time variance. Both are in ms.

    Time Domain Analysis (each value is calculated for a fix time window)
        ibi_mean - mean of heart beat variance (ibi)
        ibi_max - maximum of heart beat variance (ibi)
        ibi_min- minimum of heart beat variance (ibi)
        ibi_SDNN - standard deviation of ibi
        ibi_RMSSD - root mean square of successive differences of the ibi
    
    Time-Frequency Domain Analysis
        aVLF - spectral power of very low frequency (VLF), calculated by area under curve
        aLF - spectral power of low frequency (LF)
        aHF - spectral power of high frequency (HF)
        aTotal - aVLF+aLF+aHF
        pVLF - aVLF/aTotal
        pLF - aLF/aTotal
        pHF - aHF/aTotal
        nLF - aLF/(aLF+aHF)
        nHF - aHF/(aLF+aHF)
        LFHF - aLF/aHF
        peakVLF - peak frequency within VLF
        peakLF -peak frequency within LF
        peakHF - peak frequency within HF

RSP - Features of Spectral Power Analysis	
    Snorm_power - normalised spectral power of frequency bands
    S_indx - feature index of Snorm_power (channel, band)
    RS_power - relative spectral power across frequency bands
    smoothRS_norm - smoothed and normalised RS_power
    RS_indx - feature index of RS_power (channel1, channel2, band of channel1, band of channel2)

Phase Synchronisation Across Frequency Bands
    Calculated by Phase Entropy
        EEGphaseE - across EEG frequency bands
            phaseE - phase entropy value
            f_indx - feature index of phaseE (channel1,channel2, band of channel1, band of channel 2)
        EEGECGphaseE - across EEG frequency bands and filtered ECG signal (5-15Hz)
            phaseE - phase entropy value
            f_indx - feature index of phaseE (EEG channel, band of EEG channel, ECG channel)
        EEGIBIphaseE - across EEG and ibi frequency bands 
            phaseE - phase entropy value
            f_indx - feature index of phaseE (EEG channel, band of EEG channel, ibi channel)

    Power Phase Coupling
        EEGpowerECGphase - across EEG frequency bands and filtered ECG signal (5-15Hz)
            phase power - phase power coupling value
            f_indx - feature index of phaseE (EEG channel, band of EEG channel, ECG channel)
        EEGpowerIBIphase - across EEG and ibi frequency bands 
            phasepower - phase power coupling value
            f_indx - feature index of phaseE (EEG channel, band of EEG channel, ibi channel)
        EEGphaseECGpower - across EEG frequency bands and filtered ECG signal (5-15Hz)
            phase power - phase power coupling value
            f_indx - feature index of phaseE (EEG channel, band of EEG channel, ECG channel)
        EEGphaseIBIpower - across EEG and ibi frequency bands 
            phasepower - phase power coupling value
            f_indx - feature index of phaseE (EEG channel, band of EEG channel, ibi channel)

    Power Power Coupling
        EEGpowerECGpower - across EEG frequency bands and filtered ECG signal (5-15Hz)
            powerpower - phase power coupling value
            f_indx - feature index of phaseE (EEG channel, band of EEG channel, ECG channel)
        EEGpowerIBIpower - across EEG and ibi frequency bands 
            powerpower - phase power coupling value
            f_indx - feature index of phaseE (EEG channel, band of EEG channel, ibi channel)

Bands of EEG: Delta (0.5-4Hz), Theta (4-8Hz), Alpha (8-15Hz), Beta (15-30Hz), Gamma (30Hz until the Nyquist Frequency)
Bands of IBI: VLF(0-0.04Hz), LF(0.04-0.15Hz), HF(0.15-0.5Hz)
```

## <a name="seizure_prediction"></a>Seizure Prediction
This component is used to train a patient-specific model and perform seizure prediction. It assumes that you have already done [Data Extraction](#data_extraction) and [Feature Extraction](#feature_extraction).

### Specification files
The specification of stacked autoencoders (SAEs) are specified in `yaml` files, which are in the following directory:
 - SAEs with two layers (AE-LOGIS): `module/sae_feature/sae_2_layer`
 - SAEs with three layers (AE-AE-LOGIS): `module/sae_feature/sae_3_layer`

In each `yaml` file, you can specify:
  - A list of features used for training and prediction
  - Model parameters (i.e., parameters of stacked autoencoders (SAEs)
  - Parameters of gradient descent (used for training)

### How to run this component
1. Add `PYTHONPATH=[directory of the seizure prediction system]` into the environment variables
2. Run `python pat_[PATIENT_ID] [PREICTAL_LEN_IN_SEC] [NUM_SAE_LAYERS]`
 - You will see a progress of the training, and prediction performance printed in a terminal. You can save the prediction performance using `>` or `>>` when you run this component in a terminal.
 - `NUMBER OF SAE LAYERS` can only be either 2 or 3. If you want to specify other numbers, you have to create new yaml files. Examples of yaml files can be found in `module/sae_feature/sae_2_layer` and `module/sae_feature/sae_3_layer`

Note: The output files will be stored in either `module/sae_feature/sae_[NUM_SAE_LAYERS]_layer/pat_[PATIENT_ID]` 

### How to specify features used for training and seizure prediction
In order to change a list of features used for seizure prediction, you have to change `yaml` files and `run_sae_feature.py`:
 - For `yaml` files, specify a list of features in `list_feature`. You can check the string used to specify each feature in the `feature_extraction/base.py`.
 - For `run_sae_feature.py`, specify a list of features in `list_feature`

It is important to mention that when you change features used for seizure prediction, you also have to change the parameters of the SAEs and the gradient descent.
You can find model parameters for a certain set of features in the report.

### Description of Output
Each `module/sae_feature/sae_[NUM_SAE_LAYERS]_layer/pat_[PATIENT_ID]` folder contains:
 - `models_[PREICTAL_LEN_IN_SEC]_[LEAVEOUT_SEIZURE_IDX]_[LEAVEOUT_SEIZURE_IDX]`: trained model for each fold of the leave-one-out-cross validation 
 - `prediction_thd_[THRESHOLD]_fold_[LEAVEOUT_SEIZURE_IDX]_[LEAVEOUT_SEIZURE_IDX].png`: figure of the seizure prediction in each fold