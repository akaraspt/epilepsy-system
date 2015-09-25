import pandas as pd
import numpy as np
import os
from pylearn2.utils import serial
from theano import tensor as T
from theano import function

from pylearn2ext.chbmit import CHBMIT
from pylearn2ext.epilepsiae import Epilepsiae, EpilepsiaeTest
from pylearn2ext.base import PerformanceMetric


def gen_idx_range(src_idx, end_idx):
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


def get_range(samples):
    end_sample_idx = np.where(np.diff(samples) > 1)[0]
    if end_sample_idx.size == 0:
        if samples.size > 0:
            sample_idx = np.empty(1, dtype=object)
            sample_idx[0] = np.arange(samples.size)
        else:
            return np.empty(0)
    else:
        sample_idx = gen_idx_range(end_sample_idx, samples.size)
    sample_range = np.empty(sample_idx.size, dtype=object)
    for i in range(sample_idx.size):
        sample_range[i] = samples[sample_idx[i]]

    return sample_range


def predict(model_path, dataset, channel_thresholds):
    """
    Script to evaluate the performance with leave-one-file-out cross validation.

    Parameters
    ----------
    model_path : string
        Path to the directory to load the trained model.
    dataset : dataset object
        Dataset object.
    channel_thresholds : list
        A list of thresholds for seizure detection.

    """

    try:
        model = serial.load(model_path)
    except Exception, e:
        print model_path + "Doesn't seem to be a valid model path, got this error when trying to load it:"
        print e

    print "Setting up symbolic expressions..."

    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)
    Y = T.argmax(Y, axis=1)

    f = function([X], Y)

    # Use smallish batches to avoid running out of memory
    batch_size = dataset.batch_size
    model.set_batch_size(batch_size)

    # Dataset must be multiple of batch size of some batches will have different sizes.
    # Theano convolution requires a hard-coded batch size.
    m = dataset.X.shape[0]
    extra = (batch_size - m) % batch_size
    assert (m + extra) % batch_size == 0
    import numpy as np
    if extra > 0:
        dataset.X = np.concatenate((dataset.X, np.zeros((extra, dataset.X.shape[1]),
                                                        dtype=dataset.X.dtype)),
                                   axis=0)
    assert dataset.X.shape[0] % batch_size == 0

    print "Performing predictions..."
    y = []
    for i in xrange(dataset.X.shape[0] / batch_size):
        x_arg = dataset.X[i*batch_size:(i+1)*batch_size,:]
        if X.ndim > 2:
            x_arg = dataset.get_topological_view(x_arg)
        y.append(f(x_arg.astype(X.dtype)))
    y = np.concatenate(y)
    assert y.ndim == 1
    assert y.shape[0] == dataset.X.shape[0]

    # Discard any zero-padding that was used to give the batches uniform size
    y = y[:m]

    extra = (dataset.n_channels - y.size) % dataset.n_channels
    assert (extra + y.size) % dataset.n_channels == 0
    if extra > 0:
        y = np.append(y, np.zeros(extra))

    # Reshape
    y = y.reshape(-1, y.shape[0] / dataset.n_channels)
    sum_y = np.sum(y, 0)

    # for threshold in channel_thresholds:
    #     print 'Channel threshold: ' + str(threshold)
    #     print np.where(sum_y >= threshold)[0]

    # Channel threshold
    performance = np.empty(channel_thresholds.size, dtype=object)
    for t_idx, threshold in enumerate(channel_thresholds):
        predict_seizure_seconds = np.where(sum_y>=threshold)[0]

        # Get seizure periods
        seizure_second_periods = get_range(dataset.seizure_seconds)

        total_seizures = seizure_second_periods.size
        tp = np.zeros(seizure_second_periods.size, dtype=int)
        latency = np.zeros(seizure_second_periods.size, dtype=int)

        for p_idx, period in enumerate(seizure_second_periods):
            # True positive and detection latency
            latency[p_idx] = -1
            detect_seconds = np.intersect1d(predict_seizure_seconds, period)
            if detect_seconds.size > 0:
                tp[p_idx] = detect_seconds[0]
                latency[p_idx] = detect_seconds[0] - period[0]

        # False positive
        data_seconds = np.arange(dataset.total_seconds, dtype=np.uint16)
        nonseizure_seconds = np.setdiff1d(data_seconds, dataset.seizure_seconds)
        fp = np.intersect1d(predict_seizure_seconds, nonseizure_seconds)
        fp_range = get_range(fp)

        # Total hours
        total_hours = int(round(dataset.total_seconds / 3600.))

        print str(threshold), ':', predict_seizure_seconds
        print ' Total hours:', total_hours
        print ' Total seizures:', total_seizures
        print ' True positive:', np.where(tp)[0].size
        print ' False positive:', fp.size, fp
        print ' False positive range:', fp_range.size, fp_range
        print ' Detection latency:', latency

        performance[t_idx] = PerformanceMetric(threshold=threshold,
                                               predict_seizure_seconds=predict_seizure_seconds,
                                               total_hours=total_hours,
                                               total_seizures=total_seizures,
                                               tp=tp,
                                               fp=fp,
                                               fp_range=fp_range,
                                               latency=latency)

    return performance

def predict_sdae_leave_one_out_epilepsiae_test(channel_thresholds, model_path, data_path):
    """
    Script to evaluate the performance with leave-one-file-out cross validation for Epilepsiae patients.

    Parameters
    ----------
    channel_thresholds : list
        A list of thresholds for seizure detection.
    model_path : string
        Path to the directory to load the trained model.
    data_path : string
        Path to the directory of the CHBMIT dataset.

    """

    patients = [1]
    seizures = {
        1: range(6)
    }

    for patient_id in patients:
        with open(os.path.join(model_path, 'sdae_epilepsiae_performance_p{0}'.format(patient_id)), 'wb') as f:
            f.write('leave_file_idx,threshold,total_hours,total_seizures,tp,fp,fp_range,mean_latency\n')

            for f_idx in seizures[patient_id]:
                leave_one_out_seizure = f_idx
                save_model_path = os.path.join(model_path, 'sdae_epilepsiae_p{0}_leave_{1}'.format(patient_id,
                                                                                                   leave_one_out_seizure))

                print 'Load model:', save_model_path

                dataset = EpilepsiaeTest(patient_id=patient_id,
                                         which_set='test',
                                         preprocessor_path=os.path.join(save_model_path, 'sdae_scaler.pkl'),
                                         data_dir=data_path,
                                         leave_one_out_seizure=leave_one_out_seizure,
                                         sample_size_second=1,
                                         batch_size=20)

                performance = predict(model_path=os.path.join(save_model_path, 'sdae_all.pkl'),
                                      dataset=dataset,
                                      channel_thresholds=channel_thresholds)

                for perf in performance:
                    considered_latency = perf.latency[perf.latency >= 0]
                    mean_latency = 0
                    if considered_latency.size > 0:
                        mean_latency = np.mean(considered_latency)
                    f.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'.format(f_idx,
                                                                       perf.threshold,
                                                                       perf.total_hours,
                                                                       perf.total_seizures,
                                                                       np.where(perf.tp)[0].size,
                                                                       perf.fp.size,
                                                                       perf.fp_range.size,
                                                                       mean_latency))


if __name__ == '__main__':

    channel_thresholds = np.asarray([1, 3, 5, 10, 15])
    data_path = '/Users/akara/Workspace/data/epilepsiae'
    model_path = '../models'

    predict_sdae_leave_one_out_epilepsiae_test(channel_thresholds=channel_thresholds,
                                               data_path=data_path,
                                               model_path=model_path)
