import datetime
import math
import numpy as np
import os
import pickle
import pandas as pd
from scipy.io import loadmat
from pylearn2.format.target_format import OneHotFormatter
from scipy.signal import butter, filtfilt
from sklearn import preprocessing
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter

class Epilepsiae(DenseDesignMatrix):

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

    def __init__(self, patient_id, which_set, preprocessor_path, data_dir, transform, window_size, batch_size,
                 specified_files=None, leave_one_out_file=None, axes=('b', 0, 1, 'c'), default_seed=0):
        """
        The Epilepsiae dataset customized for leave-one-file-out cross validation.

        Parameters
        ----------
        patient_id : int
            Patient ID.
        which_set : string
            Name used to specify which partition of the dataset to be loaded (e.g., 'train', 'valid', or 'test').
            If not specified, all data will be loaded.
        preprocessor_path : string
            File path to store the scaler for pre-processing the EEG data.
        data_dir : string
            Directory that store the source EEG data.
        transform : string
            Specify how to transform the data. ('multiple_channels' | 'single_channel')
        window_size : int
            Size of each sample.
        batch_size : int
            Size of the batch, used for zero-padding to make the the number samples dividable by the batch size.
        specified_files : dictionary
            Dictionary to specified which files are used for training, validation and testing.
        leave_one_out_file : int
            Index of the withheld file.
        axes : tuple
            axes of the DenseDesignMatrix.
        default_seed : int, optional
            Seed for random.

        For preprocessing, see more in
            https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/datasets/preprocessing.py

        For customizing dataset, see more in
            https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/scripts/icml_2013_wrepl/emotions/emotions_dataset.py

        """

        self.patient_id = patient_id
        self.data_dir = data_dir
        self.preprocessor_path = preprocessor_path
        self.window_size = window_size
        self.n_classes = 2
        self.default_seed = default_seed
        self.transform = transform
        self.specified_files = specified_files
        self.leave_one_out_file = leave_one_out_file
        self.batch_size = batch_size

        raw_X, raw_y = self._load_data(which_set=which_set)

        self.raw_X = raw_X
        self.raw_y = raw_y

        # Filter scalp channels
        scalp_channels_idx = np.empty(0, dtype=int)
        for ch in self.scalp_channel_labels:
            scalp_channels_idx = np.append(scalp_channels_idx, np.where(self.channel_labels == ch)[0][0])
        raw_X = raw_X[scalp_channels_idx, :]
        self.n_channels = scalp_channels_idx.size

        self.sample_shape = [self.window_size, 1, self.n_channels]
        self.sample_size = np.prod(self.sample_shape)

        # Preprocessing
        if which_set == 'train':
            scaler = preprocessing.StandardScaler()
            scaler = scaler.fit(raw_X.transpose())

            with open(self.preprocessor_path, 'w') as f:
                pickle.dump(scaler, f)

            scaled_X = scaler.transform(raw_X.transpose()).transpose()
        else:
            with open(self.preprocessor_path) as f:
                scaler = pickle.load(f)

            scaled_X = scaler.transform(raw_X.transpose()).transpose()

        # Transform data into format usable by the network
        if self.transform == 'multiple_channels':
            X, y, view_converter = self._transform_multi_channel_data(X=scaled_X, y=raw_y)
        elif self.transform == 'single_channel':
            X, y, view_converter = self._transform_single_channel_data(X=scaled_X, y=raw_y)
        else:
            raise Exception('Invalid transform mode.')

        # Zero-padding if the batch size is not compatible
        extra = (batch_size - X.shape[0]) % batch_size
        assert (X.shape[0] + extra) % batch_size == 0
        if extra > 0:
            X = np.concatenate((X, np.zeros((extra, X.shape[1]),
                                            dtype=float)),
                               axis=0)
            y = np.concatenate((y, np.zeros((extra, y.shape[1]),
                                            dtype=int)),
                               axis=0)
        assert X.shape[0] % batch_size == 0
        assert y.size % batch_size == 0

        # Initialize DenseDesignMatrix
        DenseDesignMatrix.__init__(self, X=X, y=y,
                                   view_converter=view_converter,
                                   axes=('b', 0, 1, 'c'))

    def _load_data(self, which_set):
        # Get seizure files
        seizure_files_df = pd.read_table(os.path.join(self.data_dir, 'RECORDS-WITH-SEIZURES.txt'), sep='\t')
        seizure_files_df['filename'] = seizure_files_df['filename'].str.replace('.data', '.mat', case=False)

        # TODO Filter seizure by patient_id
        seizure_files = np.unique(seizure_files_df['filename'].values)

        print 'Seizure files\n', seizure_files

        # Train, cv and test files
        if not (self.specified_files is None):
            train_files = seizure_files[self.specified_files['train_files']]
            cv_files = seizure_files[self.specified_files['cv_files']]
            test_files = seizure_files[self.specified_files['test_files']]
        elif not (self.leave_one_out_file is None):
            train_files = np.setdiff1d(seizure_files, seizure_files[self.leave_one_out_file])
            cv_files = seizure_files[self.leave_one_out_file:self.leave_one_out_file+1]
            test_files = seizure_files[self.leave_one_out_file:self.leave_one_out_file+1]
        else:
            np.random.seed(self.default_seed)
            permute_files = np.random.permutation(seizure_files)
            train_files = permute_files[:-2]
            cv_files = permute_files[-2:-1]
            test_files = permute_files[-1:]

        print 'Train files\n', train_files
        print 'CV files\n', cv_files
        print 'Test files\n', test_files
        print ''
        # np.random.seed(self.default_seed)
        # permute_files = np.random.permutation(seizure_files)
        # train_files = permute_files[:2]
        # cv_files = permute_files[2]
        # test_files = permute_files[3]

        if which_set == 'train':
            print("Loading training data...")
            files = train_files
        elif which_set == 'valid':
            print("Loading validation data...")
            files = cv_files
        elif which_set == 'test':
            print("Loading test data...")
            files = test_files
        else:
            raise ("Invalid set")

        print files

        sampling_rate = -1
        n_channels = -1
        X = None
        y = np.empty(0, dtype=int)
        seizure_seconds = np.empty(0, dtype=int)
        total_seconds = 0
        channel_labels = None

        for f in files:
            mat = loadmat(self.data_dir + '/' + f)

            # Number of channels
            if n_channels == -1:
                n_channels = mat['signals'].shape[0]
            assert n_channels == mat['signals'].shape[0]

            # Channel labels
            if channel_labels is None:
                channel_labels = np.asarray(mat['elec_names'][0][1:-1].split(','))
            assert np.all(channel_labels == np.asarray(mat['elec_names'][0][1:-1].split(',')))

            # Sampling rate
            if sampling_rate == -1:
                sampling_rate = mat['sample_freq'][0, 0]
            assert sampling_rate == mat['sample_freq'][0, 0]

            # Start time of this file
            start_ts = datetime.datetime.strptime(mat['start_ts'][0], '%Y-%m-%d %H:%M:%S.%f')

            # EEG data
            if X is None:
                X = mat['signals']
            else:
                X = np.concatenate((X, mat['signals']), axis=1)

            # Seizure labels
            # y = np.append(y, mat['y'][0, :])
            # Get labels
            match_files = seizure_files_df[seizure_files_df['filename'].str.contains(f)]
            _temp_labels = np.zeros(mat['signals'].shape[1], dtype=int)
            if match_files.shape[0] > 0:
                for index, row in match_files.iterrows():
                    start_seizure_idx = row['onset_sample']
                    end_seizure_idx = row['offset_sample']

                    _temp_labels[start_seizure_idx:end_seizure_idx+1] = 1

                    # Store index of seizure seconds
                    start_seizure_second_idx = start_seizure_idx / (sampling_rate * 1.0)
                    end_seizure_second_idx = end_seizure_idx / (sampling_rate * 1.0)
                    _temp_seizure_seconds = np.arange(int(math.floor(start_seizure_second_idx)),
                                                      int(math.ceil(end_seizure_second_idx)) + 1) # Plus 1 due to the nature of np.arange
                    seizure_seconds = np.append(seizure_seconds, _temp_seizure_seconds + total_seconds)

                    # Debugging
                    start_seizure_ts = start_ts + datetime.timedelta(seconds=_temp_seizure_seconds[0])
                    end_seizure_ts = start_ts + datetime.timedelta(seconds=_temp_seizure_seconds[-1])
                    print 'Seizure ts:', row['onset'], row['offset']
                    print 'Seizure ts (round):', start_seizure_ts, end_seizure_ts
                    print 'Seizure samples:', start_seizure_idx, end_seizure_idx
                    print 'Seizure second index:', _temp_seizure_seconds[0], _temp_seizure_seconds[-1]

            # Seizure labels
            y = np.append(y, _temp_labels)

            # Collect total seconds
            total_seconds = total_seconds + (mat['signals'].shape[1] / (sampling_rate * 1.0))

        assert total_seconds == X.shape[1] / sampling_rate

        # Zero-padding if the window size is not compatible
        extra = (self.window_size - X.shape[1]) % self.window_size
        assert (X.shape[1] + extra) % self.window_size == 0
        if extra > 0:
            X = np.concatenate((X, np.zeros((X.shape[0], extra),
                                            dtype=float)),
                               axis=1)
            y = np.append(y, np.zeros(extra, dtype=int))
        assert X.shape[1] % self.window_size == 0
        assert y.size % self.window_size == 0

        # Store metadata
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.seizure_seconds = seizure_seconds
        self.total_seconds = total_seconds
        self.channel_labels = channel_labels

        return X, y

    def _partition_data(self, X, y, partition_size):
        partition_size = max(1, partition_size)
        X_parts = np.asarray([X[:, i:i + partition_size] for i in range(0, X.shape[1], partition_size)])
        y_parts = np.asarray([y[i:i + partition_size] for i in range(0, y.size, partition_size)])
        return X_parts, y_parts

    def _transform_multi_channel_data(self, X, y):
        # Data partitioning
        parted_X, parted_y = self._partition_data(X=X, y=y, partition_size=self.window_size)
        transposed_X = np.transpose(parted_X, [0, 2, 1])
        converted_X = np.reshape(transposed_X, (transposed_X.shape[0],
                                                transposed_X.shape[1],
                                                1,
                                                transposed_X.shape[2]))

        # Create view converter
        view_converter = DefaultViewConverter(shape=self.sample_shape,
                                              axes=('b', 0, 1, 'c'))

        # Convert data into a design matrix
        view_converted_X = view_converter.topo_view_to_design_mat(converted_X)
        assert np.all(converted_X == view_converter.design_mat_to_topo_view(view_converted_X))

        # Format the target into proper format
        sum_y = np.sum(parted_y, axis=1)
        sum_y[sum_y > 0] = 1
        one_hot_formatter = OneHotFormatter(max_labels=self.n_classes)
        hot_y = one_hot_formatter.format(sum_y)

        return view_converted_X, hot_y, view_converter

    def _transform_single_channel_data(self, X, y):
        windowed_X = np.reshape(X, (-1, self.window_size))
        windowed_y = np.reshape(y, (-1, self.window_size))

        # Format the target into proper format
        sum_y = np.sum(windowed_y, axis=1)
        sum_y[sum_y > 0] = 1

        # Duplicate the labels for all channels
        dup_y = np.tile(sum_y, self.n_channels)

        one_hot_formatter = OneHotFormatter(max_labels=self.n_classes)
        hot_y = one_hot_formatter.format(dup_y)

        return windowed_X, hot_y, None


class EpilepsiaeTest(DenseDesignMatrix):

    def __init__(self, patient_id, which_set, preprocessor_path, data_dir,
                 leave_one_out_seizure, sample_size_second, batch_size,
                 default_seed=0):
        """
        The Epilepsiae dataset customized for leave-one-seizure-out cross validation.

        Parameters
        ----------
        patient_id : int
            Patient ID.
        which_set : string
            Name used to specify which partition of the dataset to be loaded (e.g., 'train', 'valid', or 'test').
            If not specified, all data will be loaded.
        preprocessor_path : string
            File path to store the scaler for pre-processing the EEG data.
        data_dir : string
            Directory that store the source EEG data.
        leave_one_out_seizure : int
            Index of the withheld seizure.
        sample_size_second : int
            Number of seconds used to specify sample size.
        batch_size : int
            Size of the batch, used to remove a few samples to make the the number samples dividable by the batch size.
        default_seed : int, optional
            Seed for random.

        For preprocessing, see more in
            https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/datasets/preprocessing.py

        For customizing dataset, see more in
            https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/scripts/icml_2013_wrepl/emotions/emotions_dataset.py

        """

        # Load data
        files = ['rec_26402102/26402102_0003.mat',
                 'rec_26402102/26402102_0007.mat',
                 'rec_26402102/26402102_0008.mat',
                 'rec_26402102/26402102_0017.mat']
        scalp_channels = np.asarray([   u'FP1',
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
                                        u'PZ'   ])
        # Get seizure information
        seizure_info = pd.read_table(os.path.join(data_dir, 'RECORDS-WITH-SEIZURES.txt'), sep='\t')
        seizure_info['filename'] = seizure_info['filename'].str.replace('.data', '.mat', case=False)

        self.data_dir = data_dir
        self.files = files
        self.seizure_info = seizure_info
        self.filter_channels = scalp_channels
        self.default_seed = default_seed
        self.leave_one_out_seizure = leave_one_out_seizure
        self.batch_size = batch_size

        X, y, n_channels, sample_size = self.load_data(which_set, sample_size_second, batch_size, preprocessor_path)
        self.n_channels = n_channels
        self.sample_size = sample_size

        view_converter = DefaultViewConverter((1, sample_size, 1))
        view_converter.set_axes(axes=['b', 0, 1, 'c'])

        DenseDesignMatrix.__init__(self, X=X, y=y,
                                   view_converter=view_converter,
                                   axes=['b', 0, 1, 'c'])


    def load_data(self, which_set, sample_size_second, batch_size, scaler_path):
        raw_data, raw_labels, channel_labels, \
        seizure_range_idx, seizure_range_second, seizure_seconds, \
        n_channels, sample_size, sampling_rate = self.load_source_data(sample_size_second)

        self.channel_labels = channel_labels
        self.seizure_seconds_src = seizure_seconds
        self.sampling_rate = sampling_rate
        self.raw_data = raw_data

        # Generate seiuzre index (rounded to be divided by the sampling rate)
        seizure_round_sample_idx = np.empty(seizure_range_second.size, dtype=object)
        for r in range(seizure_range_second.size):
            start_idx = seizure_range_second[r][0] * sampling_rate
            end_idx = seizure_range_second[r][-1] * sampling_rate
            seizure_round_sample_idx[r] = np.arange(start_idx, end_idx)

        # Generate non-seizure index
        non_seizure_round_sample_idx = np.arange(raw_data.shape[1])
        for s_idx in seizure_round_sample_idx:
            non_seizure_round_sample_idx = np.setdiff1d(non_seizure_round_sample_idx,
                                                        s_idx)

        # Partition non-seizure data into segments
        # Then randomly choose for training, cv and test sets
        n_segments = 10
        segment_size = non_seizure_round_sample_idx.size / n_segments
        segment_size = segment_size - (segment_size % sampling_rate)
        segment_idx = np.empty(n_segments, dtype=object)
        for i in range(n_segments):
            start_segment_idx = i * segment_size
            end_segment_idx = (i+1) * segment_size
            if end_segment_idx > non_seizure_round_sample_idx.size:
                end_segment_idx = non_seizure_round_sample_idx.size
            segment_idx[i] = np.arange(start_segment_idx, end_segment_idx)

        # Select test seizure index
        test_seizure_idx = self.leave_one_out_seizure
        np.random.seed(test_seizure_idx)

        # Leave-one-out cross-validation - seizure
        n_seizures = seizure_range_idx.shape[0]
        rest_seizure_idx = np.setdiff1d(np.arange(n_seizures), test_seizure_idx)
        perm_rest_seizure_idx = np.random.permutation(rest_seizure_idx)
        train_seizure_idx = perm_rest_seizure_idx
        cv_seizure_idx = perm_rest_seizure_idx

        # Leave-one-out cross-validation - non-seizure
        n_train_segments = int(n_segments * 0.6)
        n_cv_segments = int(n_segments * 0.2)
        non_seizure_segment_idx = np.arange(n_segments)
        perm_non_seizure_segment_idx = np.random.permutation(non_seizure_segment_idx)
        train_sample_segments = perm_non_seizure_segment_idx[:n_train_segments]
        cv_sample_segments = perm_non_seizure_segment_idx[n_train_segments:n_train_segments+n_cv_segments]
        test_sample_segments = perm_non_seizure_segment_idx[n_train_segments+n_cv_segments:]
        train_sample_idx = np.empty(0, dtype=int)
        for s in train_sample_segments:
            train_sample_idx = np.append(train_sample_idx, segment_idx[s])
        cv_sample_idx = np.empty(0, dtype=int)
        for s in cv_sample_segments:
            cv_sample_idx = np.append(cv_sample_idx, segment_idx[s])
        test_sample_idx = np.empty(0, dtype=int)
        for s in test_sample_segments:
            test_sample_idx = np.append(test_sample_idx, segment_idx[s])

        print 'Segment index for train, cv and test sets:', \
              train_sample_segments, cv_sample_segments, test_sample_segments

        print 'Seizure index for train, cv and test sets:', \
              train_seizure_idx, cv_seizure_idx, [test_seizure_idx]

        if which_set == 'train':
            print("Loading training data...")
            data = raw_data[:,non_seizure_round_sample_idx[train_sample_idx]]
            labels = raw_labels[non_seizure_round_sample_idx[train_sample_idx]]
            select_seizure = train_seizure_idx
        elif which_set == 'valid':
            print("Loading validation data...")
            data = raw_data[:,non_seizure_round_sample_idx[cv_sample_idx]]
            labels = raw_labels[non_seizure_round_sample_idx[cv_sample_idx]]
            select_seizure = cv_seizure_idx
        elif which_set == 'test':
            print("Loading test data...")
            data = raw_data[:,non_seizure_round_sample_idx[test_sample_idx]]
            labels = raw_labels[non_seizure_round_sample_idx[test_sample_idx]]
            select_seizure = [test_seizure_idx]
        elif which_set == 'all':
            print("Loading all data...")
            data = raw_data
            labels = raw_labels
            select_seizure = []
        else:
            raise('Invalid set.')

        # Add seizure data
        for sz in select_seizure:
            data = np.concatenate((data, raw_data[:, seizure_round_sample_idx[sz]]), axis=1)
            labels = np.concatenate((labels, raw_labels[seizure_round_sample_idx[sz]]), axis=1)

        # No filtering

        # Preprocessing
        if which_set == 'train':
            scaler = preprocessing.StandardScaler()
            scaler = scaler.fit(data.transpose())

            with open(scaler_path, 'w') as f:
                pickle.dump(scaler, f)

            data = scaler.transform(data.transpose()).transpose()
        else:
            with open(scaler_path) as f:
                scaler = pickle.load(f)

            data = scaler.transform(data.transpose()).transpose()

        # Input transformation
        X = np.reshape(data, (-1, sample_size))
        y = np.reshape(labels, (-1, sample_size))
        y = np.sum(y, 1).transpose()
        y[y > 0] = 1

        print 'Seizure index after transform:', np.where(y)[0]
        self.seizure_seconds = np.where(y)[0]

        # Duplicate the labels for all channels
        y = np.tile(y, n_channels)

        # Format the target into proper format
        n_classes = 2
        one_hot_formatter = OneHotFormatter(max_labels=n_classes)
        y = one_hot_formatter.format(y)

        # Check batch size
        cut_off = X.shape[0] % batch_size
        if cut_off > 0:
            X = X[:-cut_off,:]
            y = y[:-cut_off,:]

        return X, y, n_channels, sample_size

    def load_source_data(self, sample_size_second):
        sampling_rate = -1
        n_channels = -1
        total_seconds = 0
        raw_data = None
        labels = np.empty(0, dtype=int)
        seizure_seconds = np.empty(0, dtype=int)
        channel_labels = None

        for f in self.files:
            mat = loadmat(self.data_dir + '/' + f)
            print 'Load data .. ' + self.data_dir + '/' + f

            all_channel_labels = np.asarray(mat['elec_names'][0][1:-1].split(','))

            if self.filter_channels is None:
                n_all_channels = mat['num_channels'][0][0]
                filter_channels_idx = np.arange(n_all_channels)
            else:
                filter_channels_idx = np.empty(0, dtype=int)
                for ch in self.filter_channels:
                    filter_channels_idx = np.append(filter_channels_idx, np.where(all_channel_labels == ch)[0][0])

            # Number of channels
            if n_channels == -1:
                n_channels = mat['signals'][filter_channels_idx,:].shape[0]
                channel_labels = all_channel_labels[filter_channels_idx]
            assert n_channels == mat['signals'][filter_channels_idx,:].shape[0]
            assert n_channels == channel_labels.shape[0]

            # Sampling rate
            if sampling_rate == -1:
                sampling_rate = mat['sample_freq'][0, 0]
            assert sampling_rate == mat['sample_freq'][0, 0]

            # Start time of this file
            start_ts = datetime.datetime.strptime(mat['start_ts'][0], '%Y-%m-%d %H:%M:%S.%f')

            # Get labels
            match_files = self.seizure_info[self.seizure_info['filename'].str.contains(f)]
            _temp_labels = np.zeros(mat['signals'][filter_channels_idx,:].shape[1], dtype=int)
            if match_files.shape[0] > 0:
                for index, row in match_files.iterrows():
                    start_seizure_idx = row['onset_sample']
                    end_seizure_idx = row['offset_sample']

                    print 'Seizure ts:', row['onset'], row['offset']
                    print 'Seizure samples:', start_seizure_idx, end_seizure_idx

                    _temp_labels[start_seizure_idx:end_seizure_idx+1] = 1

                    # Store index of seizure seconds
                    start_seizure_second_idx = (start_seizure_idx) / (sampling_rate * 1.0)
                    end_seizure_second_idx = (end_seizure_idx) / (sampling_rate * 1.0)
                    _temp_seizure_seconds = np.arange(int(math.floor(start_seizure_second_idx)),
                                                      int(math.ceil(end_seizure_second_idx)) + 1) # Plus 1 due to the nature of np.arange
                    seizure_seconds = np.append(seizure_seconds, _temp_seizure_seconds + total_seconds)

                    start_seizure_ts = start_ts + datetime.timedelta(seconds=_temp_seizure_seconds[0])
                    end_seizure_ts = start_ts + datetime.timedelta(seconds=_temp_seizure_seconds[-1])
                    print 'Floor start seizure:', start_seizure_ts
                    print 'Ceil end seizure:', end_seizure_ts

                    print 'Seizure second index:', _temp_seizure_seconds

            # Check to trim data if the window size is not compatible
            n_seconds = mat['signals'][filter_channels_idx, :].shape[1] / sampling_rate
            assert n_seconds == mat['duration_in_sec'][0, 0]
            cut_off = n_seconds % sample_size_second
            if cut_off > 0:
                _temp_data = mat['signals'][filter_channels_idx, :-(cut_off * sampling_rate)]
                _temp_labels = _temp_labels[:-(cut_off * sampling_rate)]
            else:
                _temp_data = mat['signals'][filter_channels_idx,:]

            # EEG data
            if raw_data is None:
                raw_data = _temp_data
            else:
                raw_data = np.concatenate((raw_data, _temp_data), axis=1)

            # Seizure labels
            labels = np.append(labels, _temp_labels)

            # Number of seconds
            total_seconds += n_seconds - cut_off

            print ''

        assert total_seconds == raw_data.shape[1] / sampling_rate
        sample_size = sampling_rate * sample_size_second

        self.total_seconds = total_seconds

        print 'Seizure second index (all):', seizure_seconds

        # Seizure periods - idx
        all_seizure_sample_idx = np.where(labels)[0]
        seizure_range_idx = self.get_range(all_seizure_sample_idx)
        # Seizure periods - second
        seizure_range_second = self.get_range(seizure_seconds)

        return raw_data, labels, channel_labels, \
               seizure_range_idx, seizure_range_second, seizure_seconds, \
               n_channels, sample_size, sampling_rate

    def gen_idx_range(self, src_idx, end_idx):
        # Get around indexing problem of arange()
        idx = src_idx + 1

        # Create ndarray of index
        idx_range = np.empty(idx.size + 1, dtype=object)
        for i in range(idx_range.size):
            if i == 0:
                idx_range[i] = np.arange(idx[i])
            elif i == idx_range.size - 1:
                idx_range[i] = np.arange(idx[i - 1], end_idx)
            else:
                idx_range[i] = np.arange(idx[i - 1], idx[i])
        return idx_range

    def get_range(self, samples):
        end_sample_idx = np.where(np.diff(samples) > 1)[0]
        if end_sample_idx.size == 0:
            sample_idx = np.empty(1, dtype=object)
            sample_idx[0] = np.arange(samples.size)
        else:
            sample_idx = self.gen_idx_range(end_sample_idx, samples.size)
        sample_range = np.empty(sample_idx.size, dtype=object)
        for i in range(sample_idx.size):
            sample_range[i] = samples[sample_idx[i]]

        return sample_range


class EpilepsiaeDatasetExtractor(object):

    def __init__(self, patient_id, data_dir, files, seizure_info, filter_channels=None, default_seed=0):
        self.data_dir = data_dir
        self.files = files
        self.seizure_info = seizure_info
        self.filter_channels = filter_channels
        self.default_seed = default_seed

    def gen_idx_range(self, src_idx, end_idx):
        # Get around indexing problem of arange()
        idx = src_idx + 1

        # Create ndarray of index
        idx_range = np.empty(idx.size + 1, dtype=object)
        for i in range(idx_range.size):
            if i == 0:
                idx_range[i] = np.arange(idx[i])
            elif i == idx_range.size - 1:
                idx_range[i] = np.arange(idx[i - 1], end_idx)
            else:
                idx_range[i] = np.arange(idx[i - 1], idx[i])
        return idx_range

    def get_range(self, samples):
        end_sample_idx = np.where(np.diff(samples) > 1)[0]
        if end_sample_idx.size == 0:
            sample_idx = np.empty(1, dtype=object)
            sample_idx[0] = np.arange(samples.size)
        else:
            sample_idx = self.gen_idx_range(end_sample_idx, samples.size)
        sample_range = np.empty(sample_idx.size, dtype=object)
        for i in range(sample_idx.size):
            sample_range[i] = samples[sample_idx[i]]

        return sample_range

    def load_source_data(self, sample_size_second):
        sampling_rate = -1
        n_channels = -1
        total_seconds = 0
        raw_data = None
        labels = np.empty(0, dtype=int)
        seizure_seconds = np.empty(0, dtype=int)
        channel_labels = None

        for f in self.files:
            mat = loadmat(self.data_dir + '/' + f)
            print 'Load data .. ' + self.data_dir + '/' + f

            all_channel_labels = np.asarray(mat['elec_names'][0][1:-1].split(','))

            if self.filter_channels is None:
                n_all_channels = mat['num_channels'][0][0]
                filter_channels_idx = np.arange(n_all_channels)
            else:
                filter_channels_idx = np.empty(0, dtype=int)
                for ch in self.filter_channels:
                    filter_channels_idx = np.append(filter_channels_idx, np.where(all_channel_labels == ch)[0][0])

            # Number of channels
            if n_channels == -1:
                n_channels = mat['signals'][filter_channels_idx,:].shape[0]
                channel_labels = all_channel_labels[filter_channels_idx]
            assert n_channels == mat['signals'][filter_channels_idx,:].shape[0]
            assert n_channels == channel_labels.shape[0]

            # Sampling rate
            if sampling_rate == -1:
                sampling_rate = mat['sample_freq'][0, 0]
            assert sampling_rate == mat['sample_freq'][0, 0]

            # Start time of this file
            start_ts = datetime.datetime.strptime(mat['start_ts'][0], '%Y-%m-%d %H:%M:%S.%f')

            # Get labels
            match_files = self.seizure_info[self.seizure_info['filename'].str.contains(f)]
            _temp_labels = np.zeros(mat['signals'][filter_channels_idx,:].shape[1], dtype=int)
            if match_files.shape[0] > 0:
                for index, row in match_files.iterrows():
                    start_seizure_idx = row['onset_sample']
                    end_seizure_idx = row['offset_sample']

                    print 'Seizure ts:', row['onset'], row['offset']
                    print 'Seizure samples:', start_seizure_idx, end_seizure_idx

                    _temp_labels[start_seizure_idx:end_seizure_idx+1] = 1

                    # Store index of seizure seconds
                    start_seizure_second_idx = (start_seizure_idx) / (sampling_rate * 1.0)
                    end_seizure_second_idx = (end_seizure_idx) / (sampling_rate * 1.0)
                    _temp_seizure_seconds = np.arange(int(math.floor(start_seizure_second_idx)),
                                                      int(math.ceil(end_seizure_second_idx)) + 1) # Plus 1 due to the nature of np.arange
                    seizure_seconds = np.append(seizure_seconds, _temp_seizure_seconds + total_seconds)

                    start_seizure_ts = start_ts + datetime.timedelta(seconds=_temp_seizure_seconds[0])
                    end_seizure_ts = start_ts + datetime.timedelta(seconds=_temp_seizure_seconds[-1])
                    print 'Floor start seizure:', start_seizure_ts
                    print 'Ceil end seizure:', end_seizure_ts

                    print 'Seizure second index:', _temp_seizure_seconds

            # Check to trim data if the window size is not compatible
            n_seconds = mat['signals'][filter_channels_idx, :].shape[1] / sampling_rate
            assert n_seconds == mat['duration_in_sec'][0, 0]
            cut_off = n_seconds % sample_size_second
            if cut_off > 0:
                _temp_data = mat['signals'][filter_channels_idx, :-(cut_off * sampling_rate)]
                _temp_labels = _temp_labels[:-(cut_off * sampling_rate)]
            else:
                _temp_data = mat['signals'][filter_channels_idx,:]

            # EEG data
            if raw_data is None:
                raw_data = _temp_data
            else:
                raw_data = np.concatenate((raw_data, _temp_data), axis=1)

            # Seizure labels
            labels = np.append(labels, _temp_labels)

            # Number of seconds
            total_seconds += n_seconds - cut_off

            print ''

        assert total_seconds == raw_data.shape[1] / sampling_rate
        sample_size = sampling_rate * sample_size_second

        print 'Seizure second index (all):', seizure_seconds

        # Seizure periods - idx
        all_seizure_sample_idx = np.where(labels)[0]
        seizure_range_idx = self.get_range(all_seizure_sample_idx)
        # Seizure periods - second
        seizure_range_second = self.get_range(seizure_seconds)

        return raw_data, labels, channel_labels, \
               seizure_range_idx, seizure_range_second, seizure_seconds, \
               n_channels, sample_size, sampling_rate


if __name__ == '__main__':
    # dataset = Epilepsiae(patient_id=1,
    #                      which_set='train',
    #                      preprocessor_path='../models/scaler.pkl',
    #                      data_dir='/Users/akara/Workspace/data/epilepsiae',
    #                      transform='single_channel',
    #                      window_size=256,
    #                      batch_size=20)

    # dataset = Epilepsiae(patient_id=1,
    #                      which_set='test',
    #                      preprocessor_path='../models/scaler.pkl',
    #                      data_dir='/Users/akara/Workspace/data/epilepsiae',
    #                      transform='single_channel',
    #                      leave_one_out_file=1,
    #                      window_size=256,
    #                      batch_size=20)

    dataset2 = EpilepsiaeTest(patient_id=1, which_set='train',
                              preprocessor_path='../models/sdae_scaler.pkl',
                              data_dir='/Users/akara/Workspace/data/epilepsiae',
                              sample_size_second=1,
                              leave_one_out_seizure=1,
                              batch_size=20)