import os
from optparse import OptionParser

import numpy as np

from module.sae_feature.train import train_loocv
from module.sae_feature.predict import predict_loocv
from feature_extraction.base import FeatureList
from utils.common_params import Params as params

if __name__ == '__main__':

    parser = OptionParser()
    (options, args) = parser.parse_args()

    if len(args) != 3:
        raise Exception('Number of arguments are not enough.')

    patient_id = args[0]
    data_dir = params.DATA_DIR

    preictal_sec = int(args[1])
    use_all_nonictals = True

    # Number of layers in SAE
    n_layers = int(args[2])
    module_dir = os.path.join(params.MODULE_DIR, 'sae_feature/sae_' + str(n_layers) + '_layer')

    ####################
    # Training
    ####################

    if n_layers == 2:
        # Configuration yaml files for SAE
        yamls = np.empty(n_layers + 1, dtype=object)
        yamls[0] = os.path.join(module_dir, 'ae_l1.yaml')
        yamls[1] = os.path.join(module_dir, 'softmax.yaml')
        yamls[2] = os.path.join(module_dir, 'sae.yaml')
    elif n_layers == 3:
        # Configuration yaml files for SAE
        yamls = np.empty(n_layers + 1, dtype=object)
        yamls[0] = os.path.join(module_dir, 'ae_l1.yaml')
        yamls[1] = os.path.join(module_dir, 'ae_l2.yaml')
        yamls[2] = os.path.join(module_dir, 'softmax.yaml')
        yamls[3] = os.path.join(module_dir, 'sae.yaml')
    else:
        raise Exception('There is no yaml configuration for the input number of layers.')

    train_loocv(patient_id=patient_id,
                module_dir=module_dir,
                data_dir=data_dir,
                yamls=yamls,
                preictal_sec=preictal_sec,
                use_all_nonictals=use_all_nonictals)

    print ''
    print ''
    print '=================================================================='
    print '=================================================================='
    print ''
    print ''

    ####################
    # Prediction
    ####################

    n_selected_features = -1    # Don't use feature selection
    use_available_preictal_period = True
    list_thd_firing_pow = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])

    predict_loocv(patient_id=patient_id,
                  module_dir=module_dir,
                  data_dir=data_dir,
                  list_thd_firing_pow=list_thd_firing_pow,
                  preictal_sec=preictal_sec,
                  list_features=[
                      # FeatureList.HRV_IBI_MEAN,
                      # FeatureList.HRV_IBI_SDNN,
                      # FeatureList.HRV_IBI_RMSSD,
                      # FeatureList.HRV_pVLF,
                      # FeatureList.HRV_pLF,
                      # FeatureList.HRV_pHF,
                      # FeatureList.HRV_LFHF,
                      # FeatureList.EEG_RSP_NORM_SPEC_POW,
                      FeatureList.EEG_RSP_SMOOTH_RS_NORM,
                      # FeatureList.EEG_PHASE_ENTROPY,
                      # FeatureList.EEG_ECG_PHASE_ENTROPY,
                      # FeatureList.EEG_IBI_PHASE_ENTROPY,
                      # FeatureList.EEG_POWER_ECG_PHASE,
                      # FeatureList.EEG_POWER_IBI_PHASE,
                      # FeatureList.EEG_PHASE_ECG_POWER,
                      # FeatureList.EEG_PHASE_IBI_POWER,
                      # FeatureList.EEG_POWER_ECG_POWER,
                      # FeatureList.EEG_POWER_IBI_POWER
                  ],
                  n_selected_features=n_selected_features,
                  use_all_nonictals=use_all_nonictals,
                  use_available_preictal_period=use_available_preictal_period)
