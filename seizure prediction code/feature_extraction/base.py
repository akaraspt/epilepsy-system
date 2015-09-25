class FeatureList(object):

    # Features of Heart Rate Variability Analysis
    HRV_IBI_MEAN = 'HRV.ibi_mean'
    HRV_IBI_SDNN = 'HRV.ibi_SDNN'
    HRV_IBI_RMSSD = 'HRV.ibi_RMSSD'
    HRV_pVLF = 'HRV.pVLF'
    HRV_pLF = 'HRV.pLF'
    HRV_pHF = 'HRV.pHF'
    HRV_LFHF = 'HRV.LFHF'

    # Features of Spectral Power Analysis
    EEG_RSP_NORM_SPEC_POW = 'RSP.Snorm_power'
    EEG_RSP_SMOOTH_RS_NORM = 'RSP.smoothRS_norm'

    # Phase Synchronization Across Frequency Bands (only Phase Entropy)
    EEG_PHASE_ENTROPY = 'EEGphaseE.phaseE'
    EEG_ECG_PHASE_ENTROPY = 'EEGECGphaseE.phaseE'
    EEG_IBI_PHASE_ENTROPY = 'EEGIBIphaseE.phaseE'

    # Power and Phase Coupling
    EEG_POWER_ECG_PHASE = 'EEGpowerECGphase.phasepower'
    EEG_POWER_IBI_PHASE = 'EEGpowerIBIphase.phasepower'
    EEG_PHASE_ECG_POWER = 'EEGphaseECGpower.phasepower'
    EEG_PHASE_IBI_POWER = 'EEGphaseIBIpower.phasepower'

    # Power and Power Coupling
    EEG_POWER_ECG_POWER = 'EEGpowerECGpower.powerpower'
    EEG_POWER_IBI_POWER = 'EEGpowerIBIpower.powerpower'

    @classmethod
    def get_info(cls, feature_name):
        if feature_name == FeatureList.HRV_IBI_MEAN:
            return {
                'feature': 'HRV',
                'param': 'ibi_mean'
            }
        elif feature_name == FeatureList.HRV_IBI_SDNN:
            return {
                'feature': 'HRV',
                'param': 'ibi_SDNN'
            }
        elif feature_name == FeatureList.HRV_IBI_RMSSD:
            return {
                'feature': 'HRV',
                'param': 'ibi_RMSSD'
            }
        elif feature_name == FeatureList.HRV_pVLF:
            return {
                'feature': 'HRV',
                'param': 'pVLF'
            }
        elif feature_name == FeatureList.HRV_pLF:
            return {
                'feature': 'HRV',
                'param': 'pLF'
            }
        elif feature_name == FeatureList.HRV_pHF:
            return {
                'feature': 'HRV',
                'param': 'pHF'
            }
        elif feature_name == FeatureList.HRV_LFHF:
            return {
                'feature': 'HRV',
                'param': 'LFHF'
            }

        #######################################

        elif feature_name == FeatureList.EEG_RSP_NORM_SPEC_POW:
            return {
                'feature': 'RSP',
                'param': 'Snorm_power'
            }
        elif feature_name == FeatureList.EEG_RSP_SMOOTH_RS_NORM:
            return {
                'feature': 'RSP',
                'param': 'smoothRS_norm'
            }

        #######################################

        elif feature_name == FeatureList.EEG_PHASE_ENTROPY:
            return {
                'feature': 'EEGphaseE',
                'param': 'phaseE'
            }
        elif feature_name == FeatureList.EEG_ECG_PHASE_ENTROPY:
            return {
                'feature': 'EEGECGphaseE',
                'param': 'phaseE'
            }
        elif feature_name == FeatureList.EEG_IBI_PHASE_ENTROPY:
            return {
                'feature': 'EEGIBIphaseE',
                'param': 'phaseE'
            }

        #######################################

        elif feature_name == FeatureList.EEG_POWER_ECG_PHASE:
            return {
                'feature': 'EEGpowerECGphase',
                'param': 'phasepower'
            }
        elif feature_name == FeatureList.EEG_POWER_IBI_PHASE:
            return {
                'feature': 'EEGpowerIBIphase',
                'param': 'phasepower'
            }
        elif feature_name == FeatureList.EEG_PHASE_ECG_POWER:
            return {
                'feature': 'EEGphaseECGpower',
                'param': 'phasepower'
            }
        elif feature_name == FeatureList.EEG_PHASE_IBI_POWER:
            return {
                'feature': 'EEGphaseIBIpower',
                'param': 'phasepower'
            }

        #######################################

        elif feature_name == FeatureList.EEG_POWER_ECG_POWER:
            return {
                'feature': 'EEGpowerECGpower',
                'param': 'powerpower'
            }
        elif feature_name == FeatureList.EEG_POWER_IBI_POWER:
            return {
                'feature': 'EEGpowerIBIpower',
                'param': 'powerpower'
            }
        else:
            raise Exception('Undefined feature.')