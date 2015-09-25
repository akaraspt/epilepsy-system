import os
import pickle

from pylearn2.config import yaml_parse


def train_layer1(patient_id, leave_one_out_seizure, data_path, yaml_file_path, save_model_path):
    """
    Script to pre-train the 1st autoencoder layers.

    Parameter settings are specified in yaml files.

    Parameters
    ----------
    patient_id : int
        Patient ID.
    leave_one_out_seizure : int
        Index of the withheld seizure.
    data_path : string
        Path to the directory of the database.
    yaml_file_path : string
        Path to the directory of the yaml files.
    save_model_path : string
        Path to the directory to save the trained model.

    """

    yaml = open("{0}/sdae_l1.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'patient_id': patient_id,
                    'leave_one_out_seizure': leave_one_out_seizure,
                    'window_size': 256,
                    'batch_size': 20,
                    'monitoring_batches': 5,
                    'nhid': 500,
                    'max_epochs': 20,
                    'data_path': data_path,
                    'save_path': save_model_path}
    yaml = yaml % (hyper_params)
    train = yaml_parse.load(yaml)
    train.main_loop()


def train_layer2(patient_id, leave_one_out_seizure, data_path, yaml_file_path, save_model_path):
    """
    Script to pre-train the 2nd autoencoder layers.

    Parameter settings are specified in yaml files.

    Parameters
    ----------
    patient_id : int
        Patient ID.
    leave_one_out_seizure : int
        Index of the withheld seizure.
    data_path : string
        Path to the directory of the database.
    yaml_file_path : string
        Path to the directory of the yaml files.
    save_model_path : string
        Path to the directory to save the trained model.

    """

    yaml = open("{0}/sdae_l2.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'patient_id': patient_id,
                    'leave_one_out_seizure': leave_one_out_seizure,
                    'window_size': 256,
                    'batch_size': 20,
                    'monitoring_batches': 5,
                    'nvis': 500,
                    'nhid': 500,
                    'max_epochs': 20,
                    'data_path': data_path,
                    'save_path': save_model_path}
    yaml = yaml % (hyper_params)
    train = yaml_parse.load(yaml)
    train.main_loop()

def train_layer3(patient_id, leave_one_out_seizure, data_path, yaml_file_path, save_model_path):
    """
    Script to pre-train the softmax layers.

    Parameter settings are specified in yaml files.

    Parameters
    ----------
    patient_id : int
        Patient ID.
    leave_one_out_seizure : int
        Index of the withheld seizure.
    data_path : string
        Path to the directory of the database.
    yaml_file_path : string
        Path to the directory of the yaml files.
    save_model_path : string
        Path to the directory to save the trained model.

    """

    yaml = open("{0}/sdae_l3.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'patient_id': patient_id,
                    'leave_one_out_seizure': leave_one_out_seizure,
                    'window_size': 256,
                    'batch_size': 20,
                    'monitoring_batches': 5,
                    'nvis': 500,
                    'n_classes': 2,
                    'max_epochs': 20,
                    'data_path': data_path,
                    'save_path': save_model_path}
    yaml = yaml % (hyper_params)
    train = yaml_parse.load(yaml)
    train.main_loop()


def train_mlp(patient_id, leave_one_out_seizure, data_path, yaml_file_path, save_model_path):
    """
    Script to fine-tune the pre-trained layers together.

    Parameter settings are specified in yaml files.

    Parameters
    ----------
    patient_id : int
        Patient ID.
    leave_one_out_seizure : int
        Index of the withheld seizure.
    data_path : string
        Path to the directory of the database.
    yaml_file_path : string
        Path to the directory of the yaml files.
    save_model_path : string
        Path to the directory to save the trained model.

    """

    yaml = open("{0}/sdae_mlp.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'patient_id': patient_id,
                    'leave_one_out_seizure': leave_one_out_seizure,
                    'window_size': 256,
                    'batch_size': 20,
                    'monitoring_batches': 5,
                    'n_classes': 2,
                    'max_epochs': 100,
                    'data_path': data_path,
                    'save_path': save_model_path}
    yaml = yaml % (hyper_params)
    train = yaml_parse.load(yaml)

    with open('{0}/sdae_l3.pkl'.format(save_model_path)) as fp:
        sdae_3 = pickle.load(fp)
        train.model.layers[-1].set_weights(sdae_3.layers[0].get_weights())
        train.model.layers[-1].set_biases(sdae_3.layers[0].get_biases())

    train.main_loop()


def train_sdae(patient_id, leave_one_out_seizure, data_path, yaml_file_path, save_model_path):
    """
    Script to run two-step training for the stacked of two autoencoders and one softmax layer.
    This script evaluate the performance of the model with leave-one-file-out cross validation.

    Firstly, it performs the pre-training one layer at a time.
    It then performs fine-tuning.

    Parameter settings are specified in yaml files.

    Parameters
    ----------
    patient_id : int
        Patient ID.
    leave_one_out_seizure : int
        Index of the withheld seizure.
    data_path : string
        Path to the directory of the database.
    yaml_file_path : string
        Path to the directory of the yaml files.
    save_model_path : string
        Path to the directory to save the trained model.

    """

    if not os.path.isdir(save_model_path):
        os.makedirs(save_model_path)

    train_layer1(patient_id, leave_one_out_seizure, data_path, yaml_file_path, save_model_path)
    train_layer2(patient_id, leave_one_out_seizure, data_path, yaml_file_path, save_model_path)
    train_layer3(patient_id, leave_one_out_seizure, data_path, yaml_file_path, save_model_path)
    train_mlp(patient_id, leave_one_out_seizure, data_path, yaml_file_path, save_model_path)


def train_sdae_leave_one_out_epilepsiae(data_path, yaml_file_path, save_model_path):
    """
    Script to run leave-one-file-out cross validation for Epilepsiae patients.

    Parameters
    ----------
    data_path : string
        Path to the directory of the database.
    yaml_file_path : string
        Path to the directory of the yaml files.
    save_model_path : string
        Path to the directory to save the trained model.

    """

    patients = [1]
    files = {
        1: range(6)
    }

    for patient_id in patients:

        print ''
        print '======== [PATIENT ' + str(patient_id) + '] ========'
        print ''

        for f_idx in files[patient_id]:
            print ''
            print '----[ Leave ' + str(f_idx) + ' out ]----'
            print ''
            leave_one_out_seizure = f_idx
            train_sdae(patient_id=patient_id,
                       leave_one_out_seizure=leave_one_out_seizure,
                       yaml_file_path=yaml_file_path,
                       data_path=data_path,
                       save_model_path=os.path.join(save_model_path, 'sdae_epilepsiae_p{0}_leave_{1}'.format(patient_id,
                                                                                                             leave_one_out_seizure)))


if __name__ == '__main__':
    train_sdae_leave_one_out_epilepsiae(data_path='/Users/akara/Workspace/data/epilepsiae',
                                        yaml_file_path='../yaml/sdae_epilepsiae_test',
                                        save_model_path='../models')