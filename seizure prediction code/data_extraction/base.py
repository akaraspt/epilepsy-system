import numpy as np

from abc import ABCMeta, abstractmethod
from pylearn2.datasets import DenseDesignMatrix


class DatasetLoader(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def load_data(self):
        pass


class ModifiedDenseDesignMatrix(DenseDesignMatrix):

    def get_data(self):
        '''
        Overrides DenseDesignMatrix.get_data() to give a *different* balanced batch each time it's called.

        '''

        if self.balance_class and (self.which_set == 'train' or self.which_set == 'valid_train'):
            preictal_idx = np.where(np.argmax(self.y_full, axis=1) == 1)[0]
            nonictal_idx = np.setdiff1d(np.arange(self.y_full.shape[0]), preictal_idx)

            if self.which_set == 'train':
                # Randomly choose different sets of nonictal samples
                balance_nonictal_idx = np.random.choice(nonictal_idx,
                                                        size=preictal_idx.size,
                                                        replace=False)
            else:
                # Choose the same nonictal samples
                balance_nonictal_idx = nonictal_idx[np.arange(preictal_idx.size)]

            # Remove some nonictal samples to make the dataset size compatible with the batch size
            extra = (self.batch_size - (preictal_idx.size + balance_nonictal_idx.size)) % self.batch_size
            if extra > 0:
                n_remove_idx = self.batch_size - extra
                first_half_remove = int(n_remove_idx * 0.5)
                second_half_remove = n_remove_idx - first_half_remove
                preictal_idx = preictal_idx[first_half_remove:]
                balance_nonictal_idx = balance_nonictal_idx[second_half_remove:]

            balance_idx = np.append(preictal_idx, balance_nonictal_idx)

            print ''
            print '[' + self.which_set + ': use balance class]'
            print ' preictal: {0}, nonictal: {1}'.format(preictal_idx.size, balance_nonictal_idx.size)
            print ' preictal: {0}, nonictal: {1}'.format(preictal_idx.size / (balance_idx.size * 1.0),
                                                         balance_nonictal_idx.size / (balance_idx.size * 1.0))
            # print 'Select preictal idx:', preictal_idx
            # print 'Select nonictal idx:', balance_nonictal_idx
            print ''

            self.X = self.X_full[balance_idx, :]
            self.y = self.y_full[balance_idx, :]

            return (self.X, self.y)

        if self.y is None:
            return self.X
        else:
            preictal_idx = np.where(np.argmax(self.y, axis=1) == 1)[0]
            nonictal_idx = np.setdiff1d(np.arange(self.y.shape[0]), preictal_idx)
            print ''
            print '[' + self.which_set + ': use unbalance class]'
            print ' preictal: {0}, nonictal: {1}'.format(preictal_idx.size / (self.y.shape[0] * 1.0),
                                                         nonictal_idx.size / (self.y.shape[0] * 1.0))
            print ''
            return (self.X, self.y)


class SharedExtension(object):

    '''
    Zero-padding if the batch size is not compatible.

    '''
    def zero_pad(self, X, y, batch_size):
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

        return X, y