import os
from pylearn2ext.chbmit import CHBMIT
from pylearn2ext.epilepsiae import EpilepsiaeTest


def compute_n_samples_chbmit():
    patients = [1, 3, 5, 8, 10, 20]
    model_path = '../models'
    data_path = '/Users/akara/Workspace/data/chbmit'
    with open(os.path.join(model_path, 'sdae_chbmit_train_test_samples'), 'wb') as f:
        f.write('patient_id,leave_file_idx,total_samples,train_samples,test_samples\n')

        for patient_id in patients:
            files = {
                1: range(7),
                3: range(7),
                5: range(5),
                8: range(5),
                10: range(7),
                20: range(6),
            }

            for f_idx in files[patient_id]:
                leave_one_out_file = f_idx

                dataset = CHBMIT(patient_id=patient_id,
                                 which_set='train',
                                 preprocessor_path=os.path.join(model_path, 'sdae_scaler.pkl'),
                                 data_dir=data_path,
                                 transform='single_channel',
                                 leave_one_out_file=leave_one_out_file,
                                 window_size=256,
                                 batch_size=20)

                n_train_samples = dataset.X.shape[0]

                dataset = CHBMIT(patient_id=patient_id,
                                 which_set='test',
                                 preprocessor_path=os.path.join(model_path, 'sdae_scaler.pkl'),
                                 data_dir=data_path,
                                 transform='single_channel',
                                 leave_one_out_file=leave_one_out_file,
                                 window_size=256,
                                 batch_size=20)

                n_test_samples = dataset.X.shape[0]
                n_total_samples = n_train_samples + n_test_samples

                f.write('{0},{1},{2},{3},{4}\n'.format(patient_id, leave_one_out_file, n_total_samples, n_train_samples,
                                                       n_test_samples))


def compute_n_samples_epilepsiae():
    patients = [1]
    model_path = '../models'
    data_path = '/Users/akara/Workspace/data/epilepsiae'
    with open(os.path.join(model_path, 'sdae_epilepsiae_train_test_samples'), 'wb') as f:
        f.write('patient_id,leave_seizure_idx,total_samples,train_samples,test_samples\n')

        for patient_id in patients:
            seizures = {
                1: range(6)
            }

            for s_idx in seizures[patient_id]:
                leave_one_out_seizure = s_idx

                dataset = EpilepsiaeTest(patient_id=patient_id,
                                 which_set='train',
                                 preprocessor_path=os.path.join(model_path, 'sdae_scaler.pkl'),
                                 data_dir=data_path,
                                 leave_one_out_seizure=leave_one_out_seizure,
                                 sample_size_second=1,
                                 batch_size=20)

                n_train_samples = dataset.X.shape[0]

                dataset = EpilepsiaeTest(patient_id=patient_id,
                                 which_set='test',
                                 preprocessor_path=os.path.join(model_path, 'sdae_scaler.pkl'),
                                 data_dir=data_path,
                                 leave_one_out_seizure=leave_one_out_seizure,
                                 sample_size_second=1,
                                 batch_size=20)

                n_test_samples = dataset.X.shape[0]
                n_total_samples = n_train_samples + n_test_samples

                f.write('{0},{1},{2},{3},{4}\n'.format(patient_id, leave_one_out_seizure, n_total_samples, n_train_samples,
                                                       n_test_samples))


if __name__ == '__main__':
    # compute_n_samples_chbmit()
    compute_n_samples_epilepsiae()