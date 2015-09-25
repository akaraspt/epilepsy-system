import os
import pickle

import numpy as np
from pylearn2.config import yaml_parse
from scipy.io import loadmat

from utils.common_params import Params as params


def pretrain_layer(yaml_path, hyper_params):
    yaml = open(yaml_path, 'r').read()
    yaml = yaml % (hyper_params)
    train = yaml_parse.load(yaml)
    train.main_loop()


def train_sae(yaml_path, hyper_params, pretrained_classifier_path):
    yaml = open(yaml_path, 'r').read()
    yaml = yaml % (hyper_params)
    train = yaml_parse.load(yaml)

    # Initialize parameters of the classifier layer with the pretrained weights
    # Note: pylearn2 does not support this, so this is a way to initialize with the pretrained weights
    with open(pretrained_classifier_path) as fp:
        pretrain_classifier = pickle.load(fp)
        train.model.layers[-1].set_weights(pretrain_classifier.layers[0].get_weights())
        train.model.layers[-1].set_biases(pretrain_classifier.layers[0].get_biases())

    train.main_loop()


def train_loocv(patient_id, module_dir, data_dir, yamls, preictal_sec, use_all_nonictals):
    n_layers = yamls.size - 1 # Remove the last yaml which is sae.yaml that combines all layers.

    # Print metadata for logging purpose
    print 'Preictal seconds: ' + str(preictal_sec)
    print 'Use all nonictals for resampling: ' + str(use_all_nonictals)
    print ''
    for yaml in yamls:
        print '[' + yaml + ']'
        with open(yaml, 'r') as f:
            for line in f:
                print line.rstrip('\n')
        print ''
        print ''

    # Get ictal (or seizure) information
    metadata_fname = os.path.join(os.path.join(data_dir, patient_id), 'trainset_' + str(preictal_sec) + '.mat')
    metadata_mat = loadmat(metadata_fname)
    n_seizures = metadata_mat.get('preictals').size

    # Leave one out for testing
    list_leave_out_seizure_idx_test = np.arange(n_seizures)
    # list_leave_out_seizure_idx_test = np.asarray([0,1,2,3,4,5])
    for leave_out_seizure_idx_test in list_leave_out_seizure_idx_test:

        # Leave one out for cross-validation.
        # Generate CV for finding optimal parameters for SAEs.
        # Note:
        #   - The model that has the same leave-out seizure index is used for testing
        #   - The models that have different leave-out seizure indices are used for cross-validation
        #       - i.e., select the best configuration for seizure prediction
        # list_leave_out_seizure_idx_valid = np.setdiff1d(np.arange(n_seizures), leave_out_seizure_idx_test)
        # list_leave_out_seizure_idx_valid = np.arange(n_seizures)
        # list_leave_out_seizure_idx_valid = np.asarray([0,1,2,3,4,5])
        list_leave_out_seizure_idx_valid = np.asarray([leave_out_seizure_idx_test])
        for leave_out_seizure_idx_valid in list_leave_out_seizure_idx_valid:

            save_model_dir = os.path.join(module_dir, patient_id + '/models_' + str(preictal_sec) + '_' +
                                          str(leave_out_seizure_idx_valid) + '_' +
                                          str(leave_out_seizure_idx_test))

            if not os.path.isdir(save_model_dir):
                os.makedirs(save_model_dir)

            hyper_params = {'patient_id': patient_id,
                            'leave_out_seizure_idx_valid': leave_out_seizure_idx_valid,
                            'leave_out_seizure_idx_test': leave_out_seizure_idx_test,
                            'data_dir': data_dir,
                            'preictal_sec': preictal_sec,
                            'use_all_nonictals': use_all_nonictals,
                            'save_model_dir': save_model_dir}

            # Pre-training
            for l in range(n_layers):
                print '==== LOOCV ' + str(leave_out_seizure_idx_valid) + ',' + str(leave_out_seizure_idx_test) + \
                      ' Pre-train layer ' + str(l + 1) + ' ===='
                pretrain_layer(yaml_path=yamls[l],
                               hyper_params=hyper_params)

            # Fine-tuning
            print '==== LOOCV ' + str(leave_out_seizure_idx_valid) + ',' + str(leave_out_seizure_idx_test) + \
                  ' Fine-tuning ===='
            train_sae(yaml_path=yamls[n_layers],
                      hyper_params=hyper_params,
                      pretrained_classifier_path=os.path.join(save_model_dir, 'softmax.pkl'))


def main():
    patient_id = 'pat_102'
    data_dir = params.DATA_DIR

    preictal_sec = 40 * 60
    use_all_nonictals = True

    # Number of layers in SAE
    n_layers = 2
    module_dir = os.path.join(params.MODULE_DIR, 'sae_feature/sae_' + str(n_layers) + '_layer')

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


if __name__ == '__main__':
    main()