import os
import pickle

import numpy as np
import pandas as pd
from scipy.io import loadmat
from pylearn2.format.target_format import OneHotFormatter
from sklearn import preprocessing
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

from data_extraction.base import DatasetLoader, ModifiedDenseDesignMatrix, SharedExtension
from feature_extraction.base import FeatureList
from utils.common_params import Params as params


class EpilepsiaeFeatureLoader(DatasetLoader):

    def __init__(self,
                 patient_id,
                 which_set,
                 list_features,
                 leave_out_seizure_idx_valid,
                 leave_out_seizure_idx_test,
                 data_dir,
                 preictal_sec,
                 use_all_nonictals):
        self.patient_id = patient_id
        self.which_set = which_set
        self.list_features = list_features
        self.leave_out_seizure_idx_valid = leave_out_seizure_idx_valid
        self.leave_out_seizure_idx_test = leave_out_seizure_idx_test
        self.data_dir = data_dir
        self.preictal_sec = preictal_sec
        self.use_all_nonictals = use_all_nonictals

    def load_data(self):
        # Get the directory of the patient data
        patient_dir = os.path.join(self.data_dir, self.patient_id)

        # Load metadata about dataset form MAT file
        metadata_fname = os.path.join(patient_dir, 'trainset_' + str(self.preictal_sec) + '.mat')
        metadata_mat = loadmat(metadata_fname)

        # Get number of seizures
        self.n_seizures = metadata_mat.get('ictals').size

        # Get detail of the segment
        self.sampling_rate = metadata_mat['sampling_rate'][0][0]
        self.segment_sec = metadata_mat['segment_sec'][0][0]
        self.segment_samples = self.sampling_rate * self.segment_sec

        # Get the number blocks to extend from the withheld seizure
        self.n_extended_blocks_test = metadata_mat['n_extended_blocks_test'][0][0]

        self.preictal_samples = 0
        self.nonictal_samples = 0
        self.nan_non_flat_samples = 0

        # Examples of indexing through MAT file
        # mat['nonictals'][i][0]['filename'][0][0][0][j][0]
        # mat['nonictals'][i][0]['idx'][0][0][0][j][0]
        # mat['nonictals'][i][0]['n_segments'][0][0][0][0]

        # Load shuffle data
        if self.which_set == 'train' or self.which_set == 'valid_train':

            if self.which_set == 'train':
                select_idx = np.setdiff1d(range(metadata_mat['preictals'].size),
                                          np.asarray([self.leave_out_seizure_idx_valid,
                                                      self.leave_out_seizure_idx_test]))
            else:
                select_idx = np.asarray([self.leave_out_seizure_idx_valid])

            X = None
            y = None

            if self.use_all_nonictals:
                temp_preictal_X = None
                for i in select_idx:
                    print '====== Seizure', i, '======'

                    # Pre-ictal
                    temp_X = self.load_feature(part='preictals',
                                               list_features=self.list_features,
                                               seizure_idx=i,
                                               metadata_mat=metadata_mat,
                                               patient_dir=patient_dir)

                    if not (temp_preictal_X is None):
                        temp_preictal_X = np.concatenate((temp_preictal_X, temp_X), axis=1)
                    else:
                        temp_preictal_X = temp_X

                self.preictal_samples = temp_preictal_X.shape[1]

                # Non-ictal data
                temp_nonictal_X = self.load_feature(part='nonictals_all',
                                                    list_features=self.list_features,
                                                    seizure_idx=self.leave_out_seizure_idx_test,
                                                    metadata_mat=metadata_mat,
                                                    patient_dir=patient_dir)
                X = np.concatenate((temp_preictal_X, temp_nonictal_X), axis=1)
                y = np.zeros(X.shape[1], dtype=int)
                y[range(self.preictal_samples)] = 1

                self.nonictal_samples = temp_nonictal_X.shape[1]

                print 'Preictal samples: {0}, Nonictal samples: {1}'.format(self.preictal_samples,
                                                                            self.nonictal_samples)
                if not np.all(np.arange(self.preictal_samples) == np.where(y)[0]):
                    raise Exception('There is a mismatch between the number of preictal data and labels.')

            else:
                for i in select_idx:
                    print '====== Seizure', i, '======'

                    # Non-ictal data
                    temp_nonictal_X = self.load_feature(part='nonictals',
                                                        list_features=self.list_features,
                                                        seizure_idx=i,
                                                        metadata_mat=metadata_mat,
                                                        patient_dir=patient_dir)

                    # Pre-ictal
                    temp_preictal_X = self.load_feature(part='preictals',
                                                        list_features=self.list_features,
                                                        seizure_idx=i,
                                                        metadata_mat=metadata_mat,
                                                        patient_dir=patient_dir)

                    # Concatenate preictal and nonictal data
                    temp_X = np.concatenate((temp_preictal_X, temp_nonictal_X), axis=1)
                    temp_y = np.zeros(temp_X.shape[1], dtype=int)
                    temp_y[range(temp_preictal_X.shape[1])] = 1

                    # Sanity check
                    # if not (temp_preictal_X.shape[1] == temp_nonictal_X.shape[1]):
                    #     raise Exception('Unbalanced classes.')
                    print 'Preictal samples: {0}, Nonictal samples: {1}'.format(temp_preictal_X.shape[1],
                                                                                temp_nonictal_X.shape[1])
                    if not np.all(np.arange(temp_preictal_X.shape[1]) == np.where(temp_y)[0]):
                        raise Exception('There is a mismatch between the number of preictal data and labels.')

                    self.preictal_samples = self.preictal_samples + temp_preictal_X.shape[1]
                    self.nonictal_samples = self.nonictal_samples + temp_nonictal_X.shape[1]

                    if not (X is None) and not (y is None):
                        X = np.concatenate((X, temp_X), axis=1)
                        y = np.append(y, temp_y)
                    else:
                        X = temp_X
                        y = temp_y

        # Load continuous data
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

            start_block_idx = block_idx - self.n_extended_blocks_test
            end_block_idx = block_idx + self.n_extended_blocks_test + 1

            if start_block_idx < 0:
                start_block_idx = 0
            if end_block_idx > block_df.shape[0]:
                end_block_idx = block_df.shape[0]

            select_block_idx = np.arange(start_block_idx, end_block_idx)
            filenames = block_df.filename[select_block_idx].values

            X = None
            y = None
            y_label_all = None
            ictal_labels = None
            for b_idx, fname in enumerate(filenames):
                # Name of the MAT file that stores indices of flat (i.e., false) segments
                fname_flat = fname.replace('.data', '_flat_signal_segment_idx.mat')

                # Get all good indices (i.e., remove segments of flat signals)
                flat_mat = loadmat(os.path.join(patient_dir, fname_flat))
                flat_idx = np.empty(0, dtype=int)
                for j in range(flat_mat['flat_signal_segment_idx'].shape[0]):
                    flat_idx = np.append(flat_idx, np.squeeze(flat_mat['flat_signal_segment_idx'][j][0]))
                flat_idx = flat_idx - 1 # Change from MATLAB to python index system

                n_segments = np.ceil(block_df.samples[select_block_idx[b_idx]] / (self.segment_samples * 1.0))
                all_idx = np.arange(n_segments, dtype=int)
                good_idx = np.setdiff1d(all_idx, flat_idx)

                print 'Load', self.which_set, 'data from', fname

                if good_idx.size > 0:
                    # Features with shape [n_features, n_samples]
                    temp_X = self.load_list_feature(list_features=self.list_features,
                                                    sample_idx=good_idx,
                                                    fname=fname,
                                                    patient_dir=patient_dir)

                    # If this record contains preictal data in the withheld seizures, get preictal labels
                    temp_y_withheld = self.get_labels(label_type='preictals',
                                                      filename=fname,
                                                      good_idx=good_idx,
                                                      metadata_mat=metadata_mat,
                                                      n_all_segments=n_segments,
                                                      n_data_segments=temp_X.shape[1],
                                                      select_meta_idx=select_idx)

                    # If this record contains preictal data in the selected seizures, get preictal labels
                    temp_y_select = self.get_labels(label_type='preictals',
                                                    filename=fname,
                                                    good_idx=good_idx,
                                                    metadata_mat=metadata_mat,
                                                    n_all_segments=n_segments,
                                                    n_data_segments=temp_X.shape[1])

                    # If this record contains preictal data in all seizures, get preictal labels
                    temp_y_rm = self.get_labels(label_type='all_preictals',
                                                filename=fname,
                                                good_idx=good_idx,
                                                metadata_mat=metadata_mat,
                                                n_all_segments=n_segments,
                                                n_data_segments=temp_X.shape[1])

                    tmp_preictal_withheld_idx = np.where(temp_y_withheld == 1)[0]
                    tmp_preictal_select_idx = np.where(temp_y_select == 1)[0]
                    tmp_preictal_rm_idx = np.where(temp_y_rm == 1)[0]
                    tmp_preictal_select_idx = np.setdiff1d(tmp_preictal_select_idx, tmp_preictal_withheld_idx)
                    tmp_preictal_rm_idx = np.setdiff1d(tmp_preictal_rm_idx, tmp_preictal_withheld_idx)
                    tmp_preictal_rm_idx = np.setdiff1d(tmp_preictal_rm_idx, tmp_preictal_select_idx)

                    self.preictal_samples = self.preictal_samples + np.where(temp_y_withheld == 1)[0].size
                    self.nonictal_samples = self.nonictal_samples + np.where(temp_y_withheld == 0)[0].size

                    if tmp_preictal_withheld_idx.size > 0:
                        print ' Load preictal data from the withheld seizure from this file.'
                        print ' Size:', tmp_preictal_withheld_idx.size, tmp_preictal_withheld_idx
                    if tmp_preictal_select_idx.size > 0:
                        print ' Load preictal data from selected seizures in addition to the withheld seizure from this file.'
                        print ' Size:', tmp_preictal_select_idx.size, tmp_preictal_select_idx
                    if tmp_preictal_rm_idx.size > 0:
                        print ' Load preictal data from removed seizures in addition to the withheld seizure from this file.'
                        print ' Size:', tmp_preictal_rm_idx.size, tmp_preictal_rm_idx

                    # Sanity check
                    if np.intersect1d(tmp_preictal_withheld_idx, tmp_preictal_select_idx).size > 0:
                        raise Exception('There is an overlapped of the labels between the withheld seizures, and the selected seizures.')
                    if np.intersect1d(tmp_preictal_select_idx, tmp_preictal_rm_idx).size > 0:
                        raise Exception('There is an overlapped of the labels between the selected seizures, and the removed seizures.')
                    if np.intersect1d(tmp_preictal_withheld_idx, tmp_preictal_rm_idx).size > 0:
                        raise Exception('There is an overlapped of the labels between the withheld seizures, and the removed seizures.')

                    temp_y_all = np.zeros(temp_X.shape[1], dtype=int)
                    temp_y_all[tmp_preictal_withheld_idx] = 1   # Labels for the withheld seizure
                    temp_y_all[tmp_preictal_select_idx] = 2     # Labels for the selected seizure (that is not from withheld seizures)
                    temp_y_all[tmp_preictal_rm_idx] = 3         # Labels for the removed seizure (that is not from withheld seizures)

                    # If this record contains ictal data, get ictal labels
                    temp_ictal_labels = self.get_labels(label_type='all_ictals',
                                                        filename=fname,
                                                        good_idx=good_idx,
                                                        metadata_mat=metadata_mat,
                                                        n_all_segments=n_segments,
                                                        n_data_segments=temp_X.shape[1])

                    tmp_ictal_idx = np.where(temp_ictal_labels == 1)[0]
                    if tmp_ictal_idx.size > 0:
                        print ' Ictal label:', tmp_ictal_idx.size, tmp_ictal_idx

                    # Dealing with NaN features after filtering out flat segment which occurs due to noise in the data,
                    # not from flat segments
                    nan_sample_idx = np.where(np.isnan(np.sum(temp_X, 0)))[0]
                    nan_feature_idx = np.where(np.isnan(np.sum(temp_X, 1)))[0]
                    if nan_sample_idx.size > 0 or nan_feature_idx.size > 0:
                        print self.which_set, 'contains NaN at:'
                        print ' sample_idx:', good_idx[nan_sample_idx], ' feature_idx:', nan_feature_idx
                        print ' shape before remove NaN:', temp_X.shape
                        tmp_preictal_idx = np.where(temp_y_withheld == 1)[0]
                        tmp_nonictal_idx = np.where(temp_y_withheld == 0)[0]
                        nan_preictal_sample_idx = np.intersect1d(tmp_preictal_idx, nan_sample_idx)
                        nan_nonictal_sample_idx = np.intersect1d(tmp_nonictal_idx, nan_sample_idx)
                        if nan_preictal_sample_idx.size > 0:
                            print ' NaN are in preictal index:', good_idx[nan_preictal_sample_idx]
                        if nan_nonictal_sample_idx.size > 0:
                            print ' NaN are in nonictal index:', good_idx[nan_nonictal_sample_idx]
                        all_idx = np.arange(temp_X.shape[1])
                        good_idx_1 = np.setdiff1d(all_idx, nan_sample_idx)
                        temp_X = temp_X[:, good_idx_1]
                        temp_y_all = temp_y_all[good_idx_1]
                        temp_y_withheld = temp_y_withheld[good_idx_1]
                        temp_ictal_labels = temp_ictal_labels[good_idx_1]
                        print ' shape before remove NaN:', temp_X.shape
                        self.nan_non_flat_samples = self.nan_non_flat_samples + nan_sample_idx.size

                    # Sanity check
                    tmp_nan_sample_idx = np.where(np.isnan(np.sum(temp_X, 0)))[0]
                    if tmp_nan_sample_idx.size > 0:
                        raise Exception('There is an error in removing NaN')
                    if not (temp_X.shape[1] == temp_y_all.size):
                        raise Exception('Number of feature data and labels [temp_y_all] are not equal.')
                    if not (temp_X.shape[1] == temp_y_withheld.size):
                        raise Exception('Number of feature data and labels [temp_y_withheld] are not equal.')
                    if not (temp_X.shape[1] == temp_ictal_labels.size):
                        raise Exception('Number of feature data and labels [ictal_labels] are not equal.')

                    if not (X is None) and not (y is None) and not (ictal_labels is None):
                        X = np.concatenate((X, temp_X), axis=1)
                        y = np.append(y, temp_y_withheld)
                        y_label_all = np.append(y_label_all, temp_y_all)
                        ictal_labels = np.append(ictal_labels, temp_ictal_labels)
                    else:
                        X = temp_X
                        y = temp_y_withheld
                        y_label_all = temp_y_all
                        ictal_labels = temp_ictal_labels
                else:
                    print 'There is no good segment for during this seizure'

            # Store preictal labels that are from the withheld index (use for compute accuracy), selected seizure index,
            #  and removed seizure index.
            # Note: this property will exist when which_set=='valid' or which_set=='test'
            #       as there is no need for ictal to be imported.
            self.y_label_all = y_label_all

            # Sanity check
            if np.where(y == 1)[0].size > np.where(y_label_all > 0)[0].size:
                raise Exception('There is an error in collecting preictal labels only from the leave-out-seizure index.')
            if np.where(y == 1)[0].size == np.where(y_label_all == 1)[0].size:
                print 'There is only one preictal periods, and this period is from the leave-out-seizure index.'
                if not np.all(np.where(y == 1)[0] == np.where(y_label_all == 1)[0]):
                    raise Exception('There is a mismatch between y and y_label_all.')
            if np.where(y == 1)[0].size < np.where(y_label_all > 0)[0].size:
                print 'There are more than one preictal periods.'
                if not np.all(np.where(y == 1)[0] == np.where(y_label_all == 1)[0]):
                    raise Exception('There is a mismatch between y_select_idx and y in the preictal labels of the leave-out-seizure index.')

            # Store ictal labels
            # Note: this property will exist when which_set=='valid' or which_set=='test'
            #       as there is no need for ictal to be imported.
            self.ictal_labels = ictal_labels
        else:
            raise Exception('Invalid dataset selection')

        print 'There are {0} samples that have been removed in addition to the flat signal as due to NaN.'.format(self.nan_non_flat_samples)

        X = np.transpose(X, [1, 0])
        one_hot_formatter = OneHotFormatter(max_labels=2)
        y = one_hot_formatter.format(y)

        # Sanity check
        # Note: We ignore nan_non_flat_samples if we load shuffle data as we specify the labels after the NaN have been removed
        #       In contrast to loading continuous data, we specify the labels before removing NaN, so we have to remove the NaN samples for checking
        if self.which_set == 'train' or self.which_set == 'valid_train':
            if not (X.shape[0] == self.preictal_samples + self.nonictal_samples):
                raise Exception('There is a mismatch in the number of training samples ({0} != {1}).'.format(X.shape[0],
                                                                                                             self.preictal_samples + self.nonictal_samples))
            if not (np.where(np.argmax(y, axis=1) == 1)[0].size == self.preictal_samples):
                raise Exception('There is a mismatch in the number of preictal samples and its labels ({0} != {1}).'.format(np.where(np.argmax(y, axis=1) == 1)[0].size,
                                                                                                                            self.preictal_samples))
            if not (X.shape[0] == y.shape[0]):
                raise Exception('There is a mismatch in the number of training samples and its labels ({0} != {1}).'.format(X.shape[0],
                                                                                                                            y.shape[0]))
        elif self.which_set == 'valid' or self.which_set == 'test':
            if not (X.shape[0] == self.preictal_samples + self.nonictal_samples - self.nan_non_flat_samples):
                raise Exception('There is a mismatch in the number of training samples ({0} != {1}).'.format(X.shape[0],
                                                                                                             self.preictal_samples + self.nonictal_samples - self.nan_non_flat_samples))
            if not ((np.where(np.argmax(y, axis=1) == 1)[0].size + np.where(np.argmax(y, axis=1) == 0)[0].size) ==
                        self.preictal_samples + self.nonictal_samples - self.nan_non_flat_samples):
                raise Exception('There is a mismatch in the number of samples and its labels ({0} != {1}).'.format(np.where(np.argmax(y, axis=1) == 1)[0].size + np.where(np.argmax(y, axis=1) == 0)[0].size,
                                                                                                                   self.preictal_samples))
            if not (X.shape[0] == y.shape[0]):
                raise Exception('There is a mismatch in the number of training samples and its labels ({0} != {1}).'.format(X.shape[0],
                                                                                                                            y.shape[0]))

        return X, y

    def get_labels(self, label_type, filename, good_idx, metadata_mat, n_all_segments, n_data_segments, select_meta_idx=None):
        list_meta_idx = np.empty(0, dtype=int)
        list_fname_idx = np.empty(0, dtype=int)

        # Find match preictal indices
        for meta_idx, meta in enumerate(metadata_mat[label_type]):
            for fname_idx, fname in enumerate(meta[0]['filename'][0][0][0]):
                if select_meta_idx is None:
                    if fname == filename:
                        list_meta_idx = np.append(list_meta_idx, meta_idx)
                        list_fname_idx = np.append(list_fname_idx, fname_idx)
                else:
                    if fname == filename and meta_idx == select_meta_idx:
                        list_meta_idx = np.append(list_meta_idx, meta_idx)
                        list_fname_idx = np.append(list_fname_idx, fname_idx)

        # Set labels according to the preictal indices
        if list_meta_idx.size > 0 and list_fname_idx.size > 0:
            labels = np.zeros(n_all_segments, dtype=int)
            for match_idx in range(list_meta_idx.size):

                match_meta_idx = list_meta_idx[match_idx]
                match_fname_idx = list_fname_idx[match_idx]

                # Preictal indices
                label_idx = metadata_mat[label_type][match_meta_idx][0]['idx'][0][0][0][match_fname_idx][0]
                label_idx = label_idx - 1 # Change from MATLAB to python index system

                # Set labels
                labels[label_idx] = 1

            # Sanity check
            if not (label_idx.size == np.intersect1d(good_idx, label_idx).size):
                raise Exception('Good indices and ' + label_type  + ' indices are mismatch.')

            # Remove segment of flat signals from labels
            labels = labels[good_idx]
        else:
            labels = np.zeros(n_data_segments, dtype=int)

        return labels

    def load_feature(self, part, list_features, seizure_idx, metadata_mat, patient_dir):

        X = None
        for j in range(metadata_mat[part][seizure_idx][0]['filename'][0][0][0].size):
            # Filename
            fname = metadata_mat[part][seizure_idx][0]['filename'][0][0][0][j][0]

            # Selected indices
            select_idx = metadata_mat[part][seizure_idx][0]['idx'][0][0][0][j][0]
            select_idx = select_idx - 1  # Change from MATLAB to python index system

            print 'Load', part, 'data from', fname, ':', select_idx.size

            if select_idx.size > 0:
                # Features with shape [n_features, n_samples]
                feature_X = self.load_list_feature(list_features=list_features,
                                                   sample_idx=select_idx,
                                                   fname=fname,
                                                   patient_dir=patient_dir)

                # Dealing with NaN features after filtering out flat segment which occurs due to noise in the data,
                # not from flat segments
                nan_sample_idx = np.where(np.isnan(np.sum(feature_X, 0)))[0]
                nan_feature_idx = np.where(np.isnan(np.sum(feature_X, 1)))[0]
                if nan_sample_idx.size > 0 or nan_feature_idx.size > 0:
                    print part, 'contains NaN at:'
                    print ' sample_idx:', select_idx[nan_sample_idx], ' feature_idx:', nan_feature_idx
                    print ' shape before remove NaN:', feature_X.shape
                    all_idx = np.arange(feature_X.shape[1])
                    good_idx = np.setdiff1d(all_idx, nan_sample_idx)
                    feature_X = feature_X[:, good_idx]
                    print ' shape after remove NaN:', feature_X.shape
                    self.nan_non_flat_samples = self.nan_non_flat_samples + nan_sample_idx.size

                # Sanity check
                tmp_nan_sample_idx = np.where(np.isnan(np.sum(feature_X, 0)))[0]
                if tmp_nan_sample_idx.size > 0:
                    raise Exception('There is an error in removing NaN')

                if not (X is None):
                    X = np.concatenate((X, feature_X), axis=1)
                else:
                    X = feature_X

        return X

    def load_list_feature(self, list_features, sample_idx, fname, patient_dir):

        feature_X = None
        for f_idx, feature in enumerate(list_features):
            # Name of the MAT files that store features
            fname_feature = fname.replace('.data', 'Features.mat')

            # Load features
            feature_mat = loadmat(os.path.join(patient_dir, fname_feature))

            # Feature data
            # Note: all features are with shape [n_features, n_samples]
            feature_data = feature_mat['output'][feature['feature']][0][0][feature['param']][0][0]

            # Add number of features
            feature['n_features'] = feature_data.shape[0]

            # Merge several features into one matrix
            if f_idx == 0:
                feature_X = feature_data[:, sample_idx]
            else:
                feature_X = np.concatenate((feature_X, feature_data[:, sample_idx]), axis=0)

        return feature_X


class EpilepsiaeFeatureLoaderSAE(EpilepsiaeFeatureLoader,
                                 ModifiedDenseDesignMatrix,
                                 SharedExtension):

    def __init__(self,
                 patient_id,
                 which_set,
                 list_features,
                 leave_out_seizure_idx_valid,
                 leave_out_seizure_idx_test,
                 data_dir,
                 preictal_sec,
                 use_all_nonictals,
                 preprocessor_dir,
                 n_selected_features=-1,
                 batch_size=None,
                 balance_class=True,
                 axes=('b', 0, 1, 'c'),
                 default_seed=0):

        self.balance_class = balance_class
        self.batch_size = batch_size

        tmp_list_features = np.empty(len(list_features), dtype=object)
        for f_idx in range(len(list_features)):
            tmp_list_features[f_idx] = FeatureList.get_info(list_features[f_idx])
        list_features = tmp_list_features

        print 'List of features:'
        for f in list_features:
            print f['feature'] + '.' + f['param']
        print ''

        EpilepsiaeFeatureLoader.__init__(self,
                                         patient_id=patient_id,
                                         which_set=which_set,
                                         list_features=list_features,
                                         leave_out_seizure_idx_valid=leave_out_seizure_idx_valid,
                                         leave_out_seizure_idx_test=leave_out_seizure_idx_test,
                                         data_dir=data_dir,
                                         preictal_sec=preictal_sec,
                                         use_all_nonictals=use_all_nonictals)
        # Row: samples, Col: features
        raw_X, y = self.load_data()

        if n_selected_features != -1:
            all_rank_df = None
            for f_idx, feature in enumerate(self.list_features):
                rank_df = pd.read_csv(os.path.join(data_dir, patient_id +
                                                 '/rank_feature_idx_' + feature['param'] + '_' +
                                                 'leaveout_' + str(leave_out_seizure_idx_valid) + '_' +
                                                 str(leave_out_seizure_idx_test) + '.txt'))
                if f_idx == 0:
                    all_rank_df = rank_df
                else:
                    offset_f_idx = 0
                    for i in range(f_idx):
                        offset_f_idx = offset_f_idx + self.list_features[i]['n_features']
                    rank_df['feature_idx'] = rank_df['feature_idx'].values + offset_f_idx
                    all_rank_df = pd.concat([all_rank_df, rank_df])

            sorted_feature_df = all_rank_df.sort(['D_ADH'], ascending=[0])
            self.selected_feature_idx = sorted_feature_df['feature_idx'][:n_selected_features]
            raw_X = raw_X[:, self.selected_feature_idx]
        else:
            self.selected_feature_idx = np.arange(raw_X.shape[1])

        # Print shape of input data
        print '------------------------------'
        print 'Dataset: {0}'.format(self.which_set)
        print 'Number of samples: {0}'.format(raw_X.shape[0])
        print ' Preictal samples: {0}'.format(self.preictal_samples)
        print ' Nonictal samples: {0}'.format(self.nonictal_samples)
        print ' NaN samples: {0}'.format(self.nan_non_flat_samples)
        print ' Note for ''train'' and ''valid_train'': number of samples will be equal without removing the nan samples.'
        print 'Number of features: {0}'.format(raw_X.shape[1])
        print '------------------------------'

        # Preprocessing
        if which_set == 'train':
            scaler = preprocessing.StandardScaler()
            # scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            scaler = scaler.fit(raw_X)

            with open(os.path.join(preprocessor_dir, self.patient_id + '_scaler_feature_' +
                                                     str(self.leave_out_seizure_idx_valid) + '_' +
                                                     str(self.leave_out_seizure_idx_test) + '.pkl'), 'wb') as f:
                pickle.dump(scaler, f)

            preprocessed_X = scaler.transform(raw_X)
        else:
            with open(os.path.join(preprocessor_dir, self.patient_id + '_scaler_feature_' +
                                                     str(self.leave_out_seizure_idx_valid) + '_' +
                                                     str(self.leave_out_seizure_idx_test) + '.pkl'), 'rb') as f:
                scaler = pickle.load(f)

            preprocessed_X = scaler.transform(raw_X)

        raw_X = None

        if self.which_set == 'train' or self.which_set == 'valid_train':
            # Shuffle the data
            print ''
            print '*** Shuffle data ***'
            print ''
            permute_idx = np.random.permutation(preprocessed_X.shape[0])
            preprocessed_X = preprocessed_X[permute_idx, :]
            y = y[permute_idx, :]

        if self.balance_class and (self.which_set == 'train' or self.which_set == 'valid_train'):
            self.X_full = preprocessed_X
            self.y_full = y

            (X, y) = self.get_data()
        else:
            # Zero-padding (if necessary)
            if not (self.batch_size is None):
                preprocessed_X, y = self.zero_pad(preprocessed_X, y, self.batch_size)

            X = preprocessed_X

        # Initialize DenseDesignMatrix
        DenseDesignMatrix.__init__(self,
                                   X=X,
                                   y=y,
                                   axes=axes)


if __name__ == '__main__':
    db = EpilepsiaeFeatureLoaderSAE(patient_id='pat_102',
                                    which_set='train',
                                    list_features=[
                                        FeatureList.HRV_IBI_MEAN,
                                        FeatureList.HRV_IBI_SDNN,
                                        FeatureList.HRV_IBI_RMSSD,
                                        FeatureList.HRV_pVLF,
                                        FeatureList.HRV_pLF,
                                        FeatureList.HRV_pHF,
                                        FeatureList.HRV_LFHF,
                                        FeatureList.EEG_RSP_NORM_SPEC_POW,
                                        FeatureList.EEG_RSP_SMOOTH_RS_NORM,
                                        FeatureList.EEG_PHASE_ENTROPY,
                                        FeatureList.EEG_ECG_PHASE_ENTROPY,
                                        FeatureList.EEG_IBI_PHASE_ENTROPY,
                                        FeatureList.EEG_POWER_ECG_PHASE,
                                        FeatureList.EEG_POWER_IBI_PHASE,
                                        FeatureList.EEG_PHASE_ECG_POWER,
                                        FeatureList.EEG_PHASE_IBI_POWER,
                                        FeatureList.EEG_POWER_ECG_POWER,
                                        FeatureList.EEG_POWER_IBI_POWER
                                    ],
                                    leave_out_seizure_idx_valid=0,
                                    leave_out_seizure_idx_test=0,
                                    data_dir=params.DATA_DIR,
                                    preictal_sec=40 * 60,
                                    use_all_nonictals=True,
                                    preprocessor_dir=os.path.join(params.SYSTEM_DIR, 'models'),
                                    # n_selected_features=10,
                                    batch_size=10)
    (X, y) = db.get_data()
    (X, y) = db.get_data()
    (X, y) = db.get_data()

    print '[Feature]'
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

