import os
import pickle
import time

import numpy as np
import pandas as pd
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.format.target_format import OneHotFormatter
from scipy.io import loadmat
from scipy.signal import firwin, filtfilt
from sklearn import preprocessing

from data_extraction.base import DatasetLoader, ModifiedDenseDesignMatrix, SharedExtension
from utils.common_params import Params as params


class EpilepsiaeEEGLoader(DatasetLoader):

    scalp_channel_labels = np.asarray([ u'FP1',
                                        u'FP2',
                                        u'F3',
                                        u'F4',
                                        u'C3',
                                        u'C4',
                                        u'P3',
                                        u'P4',
                                        u'O1',
                                        u'O2',
                                        u'F7',
                                        u'F8',
                                        u'T3',
                                        u'T4',
                                        u'T5',
                                        u'T6',
                                        u'FZ',
                                        u'CZ',
                                        u'PZ' ])

    def __init__(self,
                 patient_id,
                 which_set,
                 leave_out_seizure_idx_valid,
                 leave_out_seizure_idx_test,
                 data_dir):
        self.patient_id = patient_id
        self.which_set = which_set
        self.leave_out_seizure_idx_valid = leave_out_seizure_idx_valid
        self.leave_out_seizure_idx_test = leave_out_seizure_idx_test
        self.data_dir = data_dir

    def load_data(self):
        # Get the directory of the patient data
        patient_dir = os.path.join(self.data_dir, self.patient_id)

        # Load metadata about dataset form MAT file
        metadata_fname = os.path.join(patient_dir, 'trainset.mat')
        metadata_mat = loadmat(metadata_fname)

        # Get number of seizures
        self.n_seizures = metadata_mat.get('ictals').size

        # Get detail of the segment
        self.sampling_rate = metadata_mat['sampling_rate'][0][0]
        self.segment_sec = metadata_mat['segment_sec'][0][0]
        self.segment_samples = self.sampling_rate * self.segment_sec

        self.preictal_samples = 0
        self.nonictal_samples = 0

        # Examples of indexing through MAT file
        # mat['nonictals'][i][0]['filename'][0][0][0][j][0]
        # mat['nonictals'][i][0]['idx'][0][0][0][j][0]
        # mat['nonictals'][i][0]['n_segments'][0][0][0][0]

        # Balanced classes
        if self.which_set == 'train' or self.which_set == 'valid_train':

            if self.which_set == 'train':
                select_idx = np.setdiff1d(range(metadata_mat['preictals'].size),
                                          np.asarray([self.leave_out_seizure_idx_valid,
                                                      self.leave_out_seizure_idx_test]))
            else:
                select_idx = np.asarray([self.leave_out_seizure_idx_valid])

            X = None
            y = None
            for i in select_idx:
                print '====== Seizure', i, '======'

                # Non-ictal data
                temp_nonictal_X = self.load_segment(part='nonictals',
                                                    seizure_idx=i,
                                                    metadata_mat=metadata_mat,
                                                    patient_dir=patient_dir)

                # Pre-ictal
                temp_preictal_X = self.load_segment(part='preictals',
                                                    seizure_idx=i,
                                                    metadata_mat=metadata_mat,
                                                    patient_dir=patient_dir)

                # Concatenate preictal and nonictal data
                temp_X = np.concatenate((temp_preictal_X, temp_nonictal_X), axis=0)
                temp_y = np.zeros(temp_X.shape[0], dtype=int)
                temp_y[range(temp_preictal_X.shape[0])] = 1

                # Sanity check
                # if not (temp_preictal_X.shape[0] == temp_nonictal_X.shape[0]):
                #     raise Exception('Unbalanced classes.')
                print 'Preictal samples: {0}, Nonictal samples: {1}'.format(temp_preictal_X.shape[0],
                                                                            temp_nonictal_X.shape[0])
                if not np.all(np.arange(temp_preictal_X.shape[0]) == np.where(temp_y)[0]):
                    raise Exception('There is a mismatch between the number of preictal data and labels.')

                self.preictal_samples = self.preictal_samples + temp_preictal_X.shape[0]
                self.nonictal_samples = self.nonictal_samples + temp_nonictal_X.shape[0]

                if not (X is None) and not (y is None):
                    X = np.concatenate((X, temp_X), axis=0)
                    y = np.append(y, temp_y)
                else:
                    X = temp_X
                    y = temp_y

        # Unbalanced classes
        elif self.which_set == 'valid' or self.which_set == 'test':

            if self.which_set == 'valid':
                select_idx = self.leave_out_seizure_idx_valid
            else:
                select_idx = self.leave_out_seizure_idx_test

            print '====== Seizure', select_idx, '======'

            # Get metadata of all blocks
            block_df = pd.read_table(os.path.join(patient_dir, 'block_metadata.txt'), sep='\t')

            # Get block index of the selected seizure
            select_sz_fname = metadata_mat['preictals'][select_idx][0]['filename'][0][0][0][0][0]
            block_idx = np.where(block_df.filename == select_sz_fname)[0][0]

            n_padded_block = 2
            start_block_idx = block_idx - n_padded_block
            end_block_idx = block_idx + n_padded_block + 1

            if start_block_idx < 0:
                start_block_idx = 0
            if end_block_idx > block_df.shape[0]:
                end_block_idx = block_df.shape[0]

            select_block_idx = np.arange(start_block_idx, end_block_idx)
            filenames = block_df.filename[select_block_idx].values

            X = None
            y = None
            y_select_idx = None
            ictal_labels = None
            for b_idx, fname in enumerate(filenames):
                # Name of the MAT files that store EEG data
                data_fname = fname.replace('.data', '.mat')

                # Name of the MAT file that stores indices of flat (i.e., false) segments
                fname_flat = fname.replace('.data', '_flat_signal_segment_idx.mat')

                # Get all good indices (i.e., remove segments of flat signals)
                flat_mat = loadmat(os.path.join(patient_dir, fname_flat))
                flat_idx = np.empty(0, dtype=int)
                for j in range(flat_mat['flat_signal_segment_idx'].shape[0]):
                    flat_idx = np.append(flat_idx, np.squeeze(flat_mat['flat_signal_segment_idx'][j][0]))
                flat_idx = flat_idx - 1 # Change from MATLAB to python index system

                data_mat = loadmat(os.path.join(patient_dir, data_fname))

                if data_mat['signals'].shape[1] != block_df.samples[select_block_idx[b_idx]]:
                    raise Exception('There is a mismatch between the number samples specified in the metadata and '
                                    'the provided signal data')

                n_segments = np.ceil(data_mat['signals'].shape[1] / (self.segment_samples * 1.0))
                all_idx = np.arange(n_segments, dtype=int)
                good_idx = np.setdiff1d(all_idx, flat_idx)

                # Get indicies of scalp EEG channels
                elec_names = np.asarray([ename[0][0] for ename in data_mat['elec_names']])
                scalp_channels_idx = np.empty(0, dtype=int)
                for ch in self.scalp_channel_labels:
                    scalp_channels_idx = np.append(scalp_channels_idx, np.where(elec_names == ch)[0][0])

                print 'Load', self.which_set, 'data from', fname

                if good_idx.size > 0:
                    temp_X = None
                    for idx in range(good_idx.size):
                        g_idx = good_idx[idx]
                        start_sample_idx = np.uint32(g_idx) * self.segment_samples
                        end_sample_idx = np.uint32(g_idx+1) * self.segment_samples
                        if end_sample_idx > data_mat['signals'].shape[1]:
                            # Zero-padding if the window size is not compatible
                            extra = end_sample_idx - data_mat['signals'].shape[1]
                            assert (data_mat['signals'].shape[1] + extra) % self.segment_samples == 0
                            if extra > 0:
                                data_mat['signals'] = np.concatenate((data_mat['signals'],
                                                                      np.zeros((data_mat['signals'].shape[0], extra),
                                                                               dtype=float)),
                                                                     axis=1)
                            assert data_mat['signals'].shape[1] % self.segment_samples == 0
                        temp_sample_idx = np.arange(start_sample_idx, end_sample_idx)

                        if not (temp_X is None):
                            temp = data_mat['signals'][:, temp_sample_idx]
                            temp_X = np.concatenate((temp_X, np.asarray([temp[scalp_channels_idx, :]])),
                                                    axis=0)
                        else:
                            temp = data_mat['signals'][:, temp_sample_idx]
                            temp_X = np.asarray([temp[scalp_channels_idx, :]])

                    # If this record contains preictal data, get preictal labels
                    temp_preictal_meta_idx = -1
                    temp_preictal_fname_idx = -1
                    for preictal_meta_idx, preictal_meta in enumerate(metadata_mat['preictals']):
                        for preictal_fname_idx, preictal_fname in enumerate(preictal_meta[0]['filename'][0][0][0]):
                            if preictal_fname == fname:
                                temp_preictal_meta_idx = preictal_meta_idx
                                temp_preictal_fname_idx = preictal_fname_idx
                                break
                    if temp_preictal_meta_idx != -1 and temp_preictal_fname_idx != -1:
                        # Preictal indices
                        preictal_idx = metadata_mat['preictals'][temp_preictal_meta_idx][0]['idx'][0][0][0][temp_preictal_fname_idx][0]
                        preictal_idx = preictal_idx - 1 # Change from MATLAB to python index system

                        temp_y = np.zeros(n_segments, dtype=int)
                        temp_y[preictal_idx] = 1

                        # Sanity check
                        if not (preictal_idx.size == np.intersect1d(good_idx, preictal_idx).size):
                            raise Exception('Good indices and preictal indices are mismatch.')

                        # Remove segment of flat signals from labels
                        temp_y = temp_y[good_idx]

                        self.preictal_samples = self.preictal_samples + preictal_idx.size
                        self.nonictal_samples = self.nonictal_samples + (temp_y.size - preictal_idx.size)
                    else:
                        temp_y = np.zeros(temp_X.shape[0], dtype=int)
                        self.nonictal_samples = self.nonictal_samples + temp_y.size

                    # If this record contains preictal data of the leave-out-seizure index, get preictal labels
                    if temp_preictal_meta_idx == select_idx:
                        temp_y_select_idx = temp_y
                    else:
                        temp_y_select_idx = np.zeros(temp_X.shape[0], dtype=int)

                    # If this record contains ictal data, get ictal labels
                    temp_ictal_meta_idx = -1
                    temp_ictal_fname_idx = -1
                    for ictal_meta_idx, ictal_meta in enumerate(metadata_mat['ictals']):
                        for ictal_fname_idx, ictal_fname in enumerate(ictal_meta[0]['filename'][0][0][0]):
                            if ictal_fname == fname:
                                temp_ictal_meta_idx = ictal_meta_idx
                                temp_ictal_fname_idx = ictal_fname_idx
                                break
                    if temp_ictal_meta_idx != -1 and temp_ictal_fname_idx != -1:
                        # Ictal indices
                        ictal_idx = metadata_mat['ictals'][temp_ictal_meta_idx][0]['idx'][0][0][0][temp_ictal_fname_idx][0]
                        ictal_idx = ictal_idx - 1 # Change from MATLAB to python index system

                        temp_ictal_labels = np.zeros(n_segments, dtype=int)
                        temp_ictal_labels[ictal_idx] = 1

                        # Sanity check
                        if not (ictal_idx.size == np.intersect1d(good_idx, ictal_idx).size):
                            raise Exception('Good indices and ictal indices are mismatch.')

                        # Remove segment of flat signals from labels
                        temp_ictal_labels = temp_ictal_labels[good_idx]
                    else:
                        temp_ictal_labels = np.zeros(temp_X.shape[0], dtype=int)

                    # Sanity check
                    if not (temp_X.shape[0] == temp_y.size):
                        raise Exception('Number of feature data and labels are not equal.')
                    if not (temp_X.shape[0] == temp_ictal_labels.size):
                        raise Exception('Number of feature data and labels are not equal.')

                    if not (X is None) and not (y is None) and not (ictal_labels is None):
                        X = np.concatenate((X, temp_X), axis=0)
                        y = np.append(y, temp_y)
                        y_select_idx = np.append(y_select_idx, temp_y_select_idx)
                        ictal_labels = np.append(ictal_labels, temp_ictal_labels)
                    else:
                        X = temp_X
                        y = temp_y
                        y_select_idx = temp_y_select_idx
                        ictal_labels = temp_ictal_labels
                else:
                    print 'There is no good segment for during this seizure'

            # Store preictal labels that are from the leave-out-seizure index (use for compute accuracy)
            # Note: this property will exist when which_set=='valid' or which_set=='test'
            #       as there is no need for ictal to be imported.
            self.y_select_idx = y_select_idx

            # Sanity check
            if np.where(y_select_idx == 1)[0].size > np.where(y == 1)[0].size:
                raise Exception('There is an error in collecting preictal labels only from the leave-out-seizure index.')
            elif np.where(y_select_idx == 1)[0].size == np.where(y == 1)[0].size:
                print 'There is only one preictal periods, and this period is from the leave-out-seizure index.'
                if not np.all(np.where(y_select_idx == 1)[0] == np.where(y == 1)[0]):
                    raise Exception('There is a mismatch between y_select_idx and y.')
            elif np.where(y_select_idx == 1)[0].size < np.where(y == 1)[0].size:
                print 'There are more than one preictal periods.'
                if not np.all(np.intersect1d(np.where(y == 1)[0], np.where(y_select_idx == 1)[0]) == np.where(y_select_idx == 1)[0]):
                    raise Exception('There is a mismatch between y_select_idx and y in the preictal labels of the leave-out-seizure index.')

            # Store ictal labels
            # Note: this property will exist when which_set=='valid' or which_set=='test'
            #       as there is no need for ictal to be imported.
            self.ictal_labels = ictal_labels
        else:
            raise Exception('Invalid dataset selection')

        X = np.transpose(X, [0, 2, 1])
        one_hot_formatter = OneHotFormatter(max_labels=2)
        y = one_hot_formatter.format(y)

        # Sanity check
        if not (X.shape[0] == self.preictal_samples + self.nonictal_samples):
            raise Exception('There is a mismatch in the number of training samples.')
        if not (np.where(np.argmax(y, axis=1) == 1)[0].size == self.preictal_samples):
            raise Exception('There is a mismatch in the number of preictal samples and its labels.')
        if not (X.shape[0] == y.shape[0]):
            raise Exception('There is a mismatch in the number of training samples and its labels.')

        return X, y

    def load_segment(self, part, seizure_idx, metadata_mat, patient_dir):

        X = None
        for j in range(metadata_mat[part][seizure_idx][0]['filename'][0][0][0].size):
            data_fname = metadata_mat[part][seizure_idx][0]['filename'][0][0][0][j][0]

            print 'Load', part, 'data from', data_fname

            # Name of the MAT files that store EEG data
            data_mat_fname = data_fname.replace('.data', '.mat')

            # Selected indices
            segment_idx = metadata_mat[part][seizure_idx][0]['idx'][0][0][0][j][0]
            segment_idx = segment_idx - 1  # Change from MATLAB to python index system

            if segment_idx.size > 0:
                data_mat = loadmat(os.path.join(patient_dir, data_mat_fname))

                # Get indicies of scalp EEG channels
                elec_names = np.asarray([ename[0][0] for ename in data_mat['elec_names']])
                scalp_channels_idx = np.empty(0, dtype=int)
                for ch in self.scalp_channel_labels:
                    scalp_channels_idx = np.append(scalp_channels_idx, np.where(elec_names == ch)[0][0])

                segment_X = None
                for idx in range(segment_idx.size):
                    seg_idx = segment_idx[idx]
                    start_sample_idx = np.uint32(seg_idx) * self.segment_samples
                    end_sample_idx = np.uint32(seg_idx + 1) * self.segment_samples
                    if end_sample_idx > data_mat['signals'].shape[1]:
                        # Zero-padding if the window size is not compatible
                        extra = end_sample_idx - data_mat['signals'].shape[1]
                        assert (data_mat['signals'].shape[1] + extra) % self.segment_samples == 0
                        if extra > 0:
                            data_mat['signals'] = np.concatenate((data_mat['signals'],
                                                                  np.zeros(
                                                                      (data_mat['signals'].shape[0], extra),
                                                                      dtype=float)),
                                                                 axis=1)
                        assert data_mat['signals'].shape[1] % self.segment_samples == 0
                    temp_sample_idx = np.arange(start_sample_idx, end_sample_idx)

                    if not (segment_X is None):
                        temp = data_mat['signals'][:, temp_sample_idx]
                        segment_X = np.concatenate((segment_X, np.asarray([temp[scalp_channels_idx, :]])),
                                                   axis=0)
                    else:
                        temp = data_mat['signals'][:, temp_sample_idx]
                        segment_X = np.asarray([temp[scalp_channels_idx, :]])

                # Sanity check
                nan_idx = np.where(np.isnan(np.sum(segment_X, 0)))[0]
                if nan_idx.size > 0:
                    raise Exception('There are NaN in the EEG data.')

                if not (X is None):
                    X = np.concatenate((X, segment_X),
                                       axis=0)
                else:
                    X = segment_X

        return X


class EpilepsiaeEEGLoaderCNN(EpilepsiaeEEGLoader,
                             ModifiedDenseDesignMatrix,
                             SharedExtension):

    def __init__(self,
                 patient_id,
                 which_set,
                 leave_out_seizure_idx_valid,
                 leave_out_seizure_idx_test,
                 data_dir,
                 preprocessor_dir,
                 batch_size=None,
                 balance_class=True,
                 decompose_subbands=False,
                 axes=('b', 0, 1, 'c'),
                 default_seed=0):

        self.balance_class = balance_class
        self.batch_size = batch_size

        EpilepsiaeEEGLoader.__init__(self,
                                     patient_id=patient_id,
                                     which_set=which_set,
                                     leave_out_seizure_idx_valid=leave_out_seizure_idx_valid,
                                     leave_out_seizure_idx_test=leave_out_seizure_idx_test,
                                     data_dir=data_dir)

        print 'Load signal ...'
        t = time.time()
        # (# of segments, # of samples, # of channels)
        raw_X, y = self.load_data()
        elapsed = time.time() - t
        print(' Elapsed time: ' + str(elapsed) + ' seconds')

        # Preprocessing
        print 'Scaling signal ...'
        t = time.time()
        if which_set == 'train':
            # Reshape the data back to (number of samples x number of channels) for pre-processing
            unrolled_X = np.reshape(raw_X, (-1, self.scalp_channel_labels.size))

            scaler = preprocessing.StandardScaler()
            # scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            scaler = scaler.fit(unrolled_X)

            with open(os.path.join(preprocessor_dir, self.patient_id + '_scaler_eeg_' +
                                                     str(self.leave_out_seizure_idx_valid) + '_' +
                                                     str(self.leave_out_seizure_idx_test) + '.pkl'), 'w') as f:
                pickle.dump(scaler, f)

            scaled_X = raw_X.copy()
            for seg_idx in range(scaled_X.shape[0]):
                scaled_X[seg_idx, :, :] = scaler.transform(scaled_X[seg_idx, :, :])
        else:
            with open(os.path.join(preprocessor_dir, self.patient_id + '_scaler_eeg_' +
                                                     str(self.leave_out_seizure_idx_valid) + '_' +
                                                     str(self.leave_out_seizure_idx_test) + '.pkl')) as f:
                scaler = pickle.load(f)

            scaled_X = raw_X.copy()
            for seg_idx in range(scaled_X.shape[0]):
                scaled_X[seg_idx, :, :] = scaler.transform(scaled_X[seg_idx, :, :])
        elapsed = time.time() - t
        print(' Elapsed time: ' + str(elapsed) + ' seconds')

        raw_X = None

        if decompose_subbands:
            def bandpass_fir(data, lowcut_f, highcut_f, sampling_rate, window='hamming'):
                '''
                Bandpass filtering using a FIR filter.

                Parameters
                ----------
                data: numpy array
                    Input data with shape [n_samples, n_channels].
                :param lowcut_f:
                :param highcut_f:
                :param sampling_rate:
                :param window:
                :return:
                '''

                nyq_f = sampling_rate * 0.5
                n_taps = max(3 * (sampling_rate / (lowcut_f * 1.0)), 3 * nyq_f)   # Filter length

                # The filter length must be even if a passband includes the Nyquist frequency.
                if n_taps % 2 == 1:
                    n_taps = n_taps + 1

                taps = firwin(n_taps,
                              [lowcut_f, highcut_f],
                              nyq=nyq_f,
                              pass_zero=False,
                              window=window,
                              scale=False)

                # If the data is too short, zero-padding
                extra = (3 * taps.size) - data.shape[0]
                half_extra = int(np.ceil(extra / 2.0)) + 1
                if half_extra > 0:
                    padded_data = np.lib.pad(data,
                                             ((half_extra, half_extra), (0, 0)),
                                             'constant',
                                             constant_values=0)
                else:
                    padded_data = data

                filtered_data = filtfilt(taps, 1.0, padded_data, axis=0)

                if half_extra > 0:
                    return filtered_data[half_extra:-half_extra, :]
                else:
                    return filtered_data

            print 'Decompose EEG signals into 5 sub-bands ...'

            # Decompose EEG data in each segment in to 5 sub-bands
            preprocessed_X = np.zeros((scaled_X.shape[0],     # Number of segments
                                       scaled_X.shape[1],     # Segment samples
                                       5,                     # Number of sub-bands
                                       scaled_X.shape[2]))    # Number of channels

            t = time.time()
            for seg_idx in range(preprocessed_X.shape[0]):
                delta_X = bandpass_fir(scaled_X[seg_idx], 0.5, 4, self.sampling_rate)   # Delta 0.5-4 Hz
                theta_X = bandpass_fir(scaled_X[seg_idx], 4, 8, self.sampling_rate)     # Theta 4-8 Hz
                alpha_X = bandpass_fir(scaled_X[seg_idx], 8, 15, self.sampling_rate)    # Alpha 8-15 Hz
                beta_X = bandpass_fir(scaled_X[seg_idx], 15, 30, self.sampling_rate)    # Beta 15-30 Hz
                gamma_X = bandpass_fir(scaled_X[seg_idx], 30, (self.sampling_rate * 0.5) - 0.1, self.sampling_rate) # Gamma 30-Nyquist Hz

                for ch_idx in range(preprocessed_X.shape[3]):
                    preprocessed_X[seg_idx][:, 0, ch_idx] = delta_X[:, ch_idx]
                    preprocessed_X[seg_idx][:, 1, ch_idx] = theta_X[:, ch_idx]
                    preprocessed_X[seg_idx][:, 2, ch_idx] = alpha_X[:, ch_idx]
                    preprocessed_X[seg_idx][:, 3, ch_idx] = beta_X[:, ch_idx]
                    preprocessed_X[seg_idx][:, 4, ch_idx] = gamma_X[:, ch_idx]

                if seg_idx % 20 == 0 or seg_idx == preprocessed_X.shape[0] - 1:
                    print ' {0} segments {1} seconds ...'.format(seg_idx + 1, time.time() - t)

            elapsed = time.time() - t
            print ' Elapsed time: ' + str(elapsed) + ' seconds'

        else:
            # Reshape the preprocessed EEG data into a compatible format for CNN in pylearn2
            preprocessed_X = np.reshape(scaled_X, (scaled_X.shape[0],     # Number of segments
                                                   scaled_X.shape[1],     # Segment samples
                                                   1,                     # EEG data are time-series data (i.e., 1 dimension)
                                                   scaled_X.shape[2]))    # Number of channels

        scaled_X = None

        # Print shape of input data
        print '------------------------------'
        print 'Dataset: {0}'.format(self.which_set)
        print 'Number of samples: {0}'.format(preprocessed_X.shape[0])
        print ' Preictal samples: {0}'.format(self.preictal_samples)
        print ' Nonictal samples: {0}'.format(self.nonictal_samples)
        print 'Shape of each sample: ({0}, {1})'.format(preprocessed_X.shape[1], preprocessed_X.shape[2])
        print 'Number of channels: {0}'.format(preprocessed_X.shape[3])
        print '------------------------------'

        # Create a view converter
        view_converter = DefaultViewConverter(shape=[preprocessed_X.shape[1],     # Segment samples
                                                     preprocessed_X.shape[2],     # Number of sub-bands
                                                     preprocessed_X.shape[3]],    # Number of channels
                                              axes=('b', 0, 1, 'c'))

        # Sanity check
        view_converted_X = view_converter.topo_view_to_design_mat(preprocessed_X)
        assert np.all(preprocessed_X == view_converter.design_mat_to_topo_view(view_converted_X))

        preprocessed_X = None

        if self.balance_class and (self.which_set == 'train' or self.which_set == 'valid_train'):
            self.X_full = view_converted_X
            self.y_full = y

            (X, y) = self.get_data()
        else:
            # Zero-padding (if necessary)
            if not (self.batch_size is None):
                view_converted_X, y = self.zero_pad(view_converted_X, y, self.batch_size)

            X = view_converted_X

        # Initialize DenseDesignMatrix
        DenseDesignMatrix.__init__(self,
                                   X=X,
                                   y=y,
                                   view_converter=view_converter,
                                   axes=axes)


if __name__ == '__main__':
    db = EpilepsiaeEEGLoaderCNN(patient_id='pat_11002',
                                which_set='train',
                                leave_out_seizure_idx_valid=3,
                                leave_out_seizure_idx_test=7,
                                data_dir=params.DATA_DIR,
                                preprocessor_dir=os.path.join(params.SYSTEM_DIR, 'models'),
                                batch_size=10,
                                decompose_subbands=True)
    (X, y) = db.get_data()
    (X, y) = db.get_data()
    (X, y) = db.get_data()

    print '[EEG]'
    if db.which_set == 'train' or db.which_set == 'valid_train':
        print db.X_full.shape
        print db.y_full.shape
        print np.where(np.argmax(db.y_full, axis=1))[0].size
    print db.X.shape
    print db.y.shape
    print np.where(np.argmax(db.y, axis=1))[0].size
    print X.shape
    print y.shape
    print np.where(np.argmax(y, axis=1))[0].size
    print ''