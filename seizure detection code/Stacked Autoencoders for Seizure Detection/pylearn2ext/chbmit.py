import numpy as np
import os
import pickle
import pandas as pd
from scipy.io import loadmat
from pylearn2.format.target_format import OneHotFormatter
from scipy.signal import butter, filtfilt
from sklearn import preprocessing
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter

class CHBMIT(DenseDesignMatrix):

    # These are representative channel MATLAB index (it needs to be subtracted by 1 before using in python)
    rep_channel_matlab_idx = {
        1: np.asarray([7,8,11,13,14,21,22]),
        3: np.asarray([3,4,6,16,19]),
        5: np.asarray([4,5,7,8,11,12,17,18]),
        8: np.asarray([6,7,8,10,11,17,18]),
        10: np.asarray([2,3,19,20,21]),
        20: np.asarray([1,2,3,19,20,21,24,25,26,27,28])
    }

    def __init__(self, patient_id, which_set, preprocessor_path, data_dir, transform, window_size, batch_size,
                 specified_files=None, leave_one_out_file=None, axes=('b', 0, 1, 'c'), default_seed=0):
        """
        The CHBMIT dataset customized for leave-one-file-out cross validation.

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

        # Filter representative channels
        if not(self.rep_channel_matlab_idx.get(patient_id) is None):
            # Map the representative MATLAB index to python index
            # Also the raw_data read from the .mat file has already removed inactive channels
            # So we need to search for the match original index with MATLAB index
            # Then transfer to the python index
            self.rep_channel_python_idx = np.empty(0, dtype=int)
            for ch in self.rep_channel_matlab_idx[patient_id]:
                if ch in self.used_channel_matlab_idx:
                    ch_python_idx = np.where(ch == self.used_channel_matlab_idx)[0]
                    self.rep_channel_python_idx = np.append(self.rep_channel_python_idx, ch_python_idx)
                else:
                    raise Exception('There is no representative channel ' + str(ch) + ' in the input data.')
            assert np.all(self.used_channel_matlab_idx[self.rep_channel_python_idx] ==
                          self.rep_channel_matlab_idx[patient_id])

            raw_X = raw_X[self.rep_channel_python_idx, :]
            self.n_channels = self.rep_channel_python_idx.size

            print 'Used channel MATLAB index:', self.used_channel_matlab_idx
            print 'Representative channel MATLAB index:', self.rep_channel_matlab_idx[patient_id]
            print 'Representative channel Python index:', self.rep_channel_python_idx

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
        seizure_files_df = pd.read_table(os.path.join(self.data_dir, 'RECORDS-WITH-SEIZURES.txt'),
                                         sep=' ', names=['filename', 'period'], header=None)
        if self.patient_id <10:
            search_str = 'chb0' + str(self.patient_id)
        else:
            search_str = 'chb' + str(self.patient_id)
        seizure_files = seizure_files_df['filename'][seizure_files_df['filename'].str.contains(search_str)]
        seizure_files = seizure_files.str.replace('.edf', '_mod.mat', case=False).values

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
        used_channel_matlab_idx = None

        for f in files:
            mat = loadmat(self.data_dir + '/' + f)

            # Number of channels
            if n_channels == -1:
                n_channels = mat['X'].shape[0]
            assert n_channels == mat['X'].shape[0]

            # Channel labels
            if channel_labels is None:
                channel_labels = np.asarray([lb[0][0] for lb in mat['labels']])
            assert np.all(channel_labels == np.asarray([lb[0][0] for lb in mat['labels']]))

            # Channel index (MATLAB index, start from 1, not 0) used to filter active channels from the source files
            if used_channel_matlab_idx is None:
                used_channel_matlab_idx = mat['used_channel_idx'][0]
            assert np.all(used_channel_matlab_idx == mat['used_channel_idx'][0])

            # Sampling rate
            if sampling_rate == -1:
                sampling_rate = mat['sampling_rate'][0, 0]
            assert sampling_rate == mat['sampling_rate'][0, 0]

            # EEG data
            if X is None:
                X = mat['X']
            else:
                X = np.concatenate((X, mat['X']), axis=1)

            # Seizure labels
            y = np.append(y, mat['y'][0, :])

            # Store index of seizure seconds
            seizure_seconds = np.append(seizure_seconds, mat['seizure_second'][0, :] + total_seconds)

            # Collect total seconds
            total_seconds = total_seconds + (mat['X'].shape[1] / (sampling_rate * 1.0))

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
        self.used_channel_matlab_idx = used_channel_matlab_idx

        print 'Seizure seconds:', self.seizure_seconds

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


if __name__ == '__main__':
    dataset = CHBMIT(patient_id=1,
                     which_set='train',
                     preprocessor_path='../models/scaler.pkl',
                     data_dir='/Users/akara/Workspace/data/chbmit',
                     transform='single_channel',
                     window_size=256,
                     batch_size=20)

    # dataset = CHBMIT(patient_id=1,
    #                  which_set='train',
    #                  preprocessor_path='../models/scaler.pkl',
    #                  data_dir='/Users/akara/Workspace/data/chbmit',
    #                  transform='single_channel',
    #                  specified_files={
    #                      'train_files': np.asarray([0,1,2,3,4,5]),
    #                      'cv_files': np.asarray([6]),
    #                      'test_files': np.asarray([6])
    #                  },
    #                  window_size=256,
    #                  batch_size=20)

    # dataset = CHBMIT(patient_id=1,
    #                  which_set='train',
    #                  preprocessor_path='../models/scaler.pkl',
    #                  data_dir='/Users/akara/Workspace/data/chbmit',
    #                  transform='single_channel',
    #                  leave_one_out_file=4,
    #                  window_size=256,
    #                  batch_size=20)

    # from pylearn2ext.chbmit_eeg_dataset import ChbMitDatasetSDAE
    # dataset2 = ChbMitDatasetSDAE(patient_id=1,
    #                              which_set='train',
    #                              scaler_path='../models/scaler.pkl',
    #                              data_dir='/Users/akara/Workspace/data/chbmit',
    #                              sample_size_second=1,
    #                              batch_size=20)
    #
    # assert np.all(dataset.X == dataset2.X)