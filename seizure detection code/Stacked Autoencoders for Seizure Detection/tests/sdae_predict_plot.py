import os

from pylearn2.utils import serial
from theano import tensor as T
from theano import function

from pylearn2ext.chbmit import CHBMIT
from tests.plot_eeg import plot_eeg_predict_seizure_period


def predict_plot(model_path, dataset):
    """
    Script to perform seizure detection and plot the results.

    Parameters
    ----------
    model_path : string
        Path to the directory to load the trained model.
    data_path : dataset object
        Dataset object.

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

    plot_eeg_predict_seizure_period(X=dataset.raw_X,
                                    y=np.repeat(sum_y, dataset.sampling_rate),
                                    channel_labels=dataset.channel_labels,
                                    seizure_seconds=dataset.seizure_seconds,
                                    sampling_rate=dataset.sampling_rate,
                                    start_second=3600,
                                    end_second=3900,
                                    is_scale=True,
                                    n_X_ticks=6,
                                    channel_th_y_lim=[-1, 6],
                                    figure_width=800,
                                    figure_height=600)


if __name__ == '__main__':
    patient_id = 10
    leave_one_out_file = 4
    model_path = '../models'
    data_path = '/Users/akara/Workspace/data/chbmit'


    save_model_path = os.path.join(model_path, 'sdae_chbmit_p{0}_leave_{1}'.format(patient_id,
                                                                                   leave_one_out_file))

    dataset = CHBMIT(patient_id=patient_id,
                     which_set='test',
                     preprocessor_path=os.path.join(save_model_path, 'sdae_scaler.pkl'),
                     data_dir=data_path,
                     transform='single_channel',
                     leave_one_out_file=leave_one_out_file,
                     window_size=256,
                     batch_size=20)

    predict_plot(model_path=os.path.join(save_model_path, 'sdae_all.pkl'),
                 dataset=dataset)