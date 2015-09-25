import os

import numpy as np
import pandas as pd
from pylearn2.utils import serial
from theano import tensor as T
from theano import function
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix

from data_extraction.epilepsiae import EpilepsiaeFeatureLoaderSAE
from feature_extraction.base import FeatureList
from utils.index_helper import get_list_con_seq_idx
from utils.common_params import Params as params


def predict(model, dataset, batch_size=10):
    # Use smallish batches to avoid running out of memory
    model.set_batch_size(batch_size)

    print "Setting up symbolic expressions..."
    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)
    Y = T.argmax(Y, axis=1)
    f = function([X], Y)

    # Dataset must be multiple of batch size.
    m = dataset.X.shape[0]
    extra = (batch_size - m) % batch_size
    assert (m + extra) % batch_size == 0
    if extra > 0:
        dataset.X = np.concatenate((dataset.X, np.zeros((extra, dataset.X.shape[1]),
                                                        dtype=dataset.X.dtype)),
                                   axis=0)
    assert dataset.X.shape[0] % batch_size == 0

    # Prediction
    print "Performing predictions..."
    y = []
    for i in xrange(dataset.X.shape[0] / batch_size):
        x_arg = dataset.X[i * batch_size:(i + 1) * batch_size, :]
        if X.ndim > 2:
            x_arg = dataset.get_topological_view(x_arg)
        y.append(f(x_arg.astype(X.dtype)))
    y = np.concatenate(y)
    assert y.ndim == 1
    assert y.shape[0] == dataset.X.shape[0]

    # Discard any zero-padding that was used to give the batches uniform size
    y = y[:m]

    return y


def get_prediction_performance(y_hat,
                               y,
                               preictal_labels,
                               ictal_labels,
                               leave_out_seizure_idx_valid,
                               leave_out_seizure_idx_test,
                               figure_dir,
                               thd_firing_pow,
                               preictal_sec,
                               segment_sec,
                               use_available_preictal):

    y_true = np.where(y)[0]
    y_preictal_withheld = np.where(preictal_labels == 1)[0]
    y_preictal_select = np.where(preictal_labels == 2)[0]
    y_preictal_remove = np.where(preictal_labels == 3)[0]
    y_ictal = np.where(ictal_labels)[0]

    assert np.all(y_true == y_preictal_withheld)

    print 'n_preictal_segments: ' + str(y_true.size)

    list_con_idx_y_true = get_list_con_seq_idx(y_true)
    list_con_idx_y_preictal_select = get_list_con_seq_idx(y_preictal_select)
    list_con_idx_y_preictal_remove = get_list_con_seq_idx(y_preictal_remove)
    list_con_idx_y_ictal = get_list_con_seq_idx(y_ictal)

    # Get prediction results with decision making
    if use_available_preictal:
        n_preictal_samples = y_true.size * 1.0
    else:
        n_preictal_samples = preictal_sec / (segment_sec * 1.0)

    firing_pow = np.zeros(y_hat.size, dtype=float)
    for i in np.arange(y_hat.size):
        start_idx = (i + 1) - n_preictal_samples
        end_idx = i + 1
        if start_idx < 0:
            start_idx = 0

        firing_pow[i] = np.sum(y_hat[start_idx:end_idx]) / n_preictal_samples

    # # Further normalize the results from the decision making to [0, 1]
    # from sklearn import preprocessing
    # min_max_scaler = preprocessing.MinMaxScaler()
    # norm_firing_pow = min_max_scaler.fit_transform(firing_pow)

    # Limit raising alarms (1)
    # Even after a preictal period is passed, further alarm cannot be raised just by noticing firing_pow value above the threshold,
    # and the firing_pow must first fall below the threshold level, resetting the normal alarm generation
    y_pred = np.where(firing_pow >= thd_firing_pow)[0]
    if y_pred.size > 0:
        list_con_idx_y_pred = get_list_con_seq_idx(y_pred)
        y_pred_limit = np.zeros(y_hat.size, dtype=int)
        for i in range(list_con_idx_y_pred.size):
            y_pred_limit[y_pred[list_con_idx_y_pred[i][0]]] = 1
        all_alarms = np.where(y_pred_limit)[0]
    else:
        all_alarms = np.empty(0, dtype=int)

    # Limit raising alarms (2)
    # Generation of more alarms is inhibited for as long as a preictal period
    alarms = np.empty(0, dtype=int)
    for a_idx, alarm in enumerate(all_alarms):
        if a_idx > 0:
            if (alarm - alarms[-1]) > n_preictal_samples:
                alarms = np.append(alarms, alarm)
        else:
            alarms = np.append(alarms, alarm)
    inhibited_alarms = np.setdiff1d(all_alarms, alarms)
    assert np.all(np.union1d(alarms, inhibited_alarms) == all_alarms)

    # Remove the alarm triggered from not considered seizures FP
    rm_select_sz_alarms = np.intersect1d(alarms, y_preictal_select)
    rm_remove_sz_alarms = np.intersect1d(alarms, y_preictal_remove)
    alarms = np.setdiff1d(alarms, y_preictal_select)
    alarms = np.setdiff1d(alarms, y_preictal_remove)

    # Compute confusion matrix
    cm = confusion_matrix(y, y_hat)
    np.set_printoptions(precision=2)
    print 'Confusion matrix, without normalization'
    print cm

    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print 'Normalized confusion matrix'
    print cm_normalized

    # After applying the decision making
    tp = np.intersect1d(alarms, y_true)
    fp = np.setdiff1d(alarms, y_true)
    if tp.size > 0:
        time_before_ictal = ((y_true[-1] + 1) - tp[0]) * (segment_sec * 1.0)
    else:
        time_before_ictal = -1.0

    detected_sz = 0
    if tp.size > 0:
        detected_sz = 1

    result = pd.DataFrame({
        'n_seizures': 1,                    # Number of withheld seizures (i.e., 1 for LOOCV)
        'detected_sz': detected_sz,         # Number of detected seizures
        'fp': fp.size,                      # Number of false alarms (i.e., false positives)
        't_before_sz': time_before_ictal    # Seconds before the onset of the withheld seizures
    }, index=[0])

    print ''
    print result
    print ''

    ######################################################################
    # Plot prediction results
    ######################################################################
    import matplotlib.pyplot as plt
    fig_dpi = 80
    fig_width = 1500/fig_dpi
    fig_height = 800/fig_dpi
    fig_save_dpi = 200
    plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)

    # Highlight selected preictal periods
    with_label = True
    for i in range(list_con_idx_y_preictal_select.size):
        # Index of preictal period
        start_highlight_idx = y_preictal_select[list_con_idx_y_preictal_select[i]][0]
        end_highlight_idx = y_preictal_select[list_con_idx_y_preictal_select[i]][-1]

        if with_label:
            plt.axvspan(start_highlight_idx, end_highlight_idx + 1, ymax=1.15/1.2, color='c',
                        alpha=0.1, edgecolor='none',
                        label='Preictal (selected)')
            with_label = False
        else:
            plt.axvspan(start_highlight_idx, end_highlight_idx + 1, ymax=1.15/1.2, color='c',
                        alpha=0.1, edgecolor='none')

    # Highlight removed preictal periods
    with_label = True
    for i in range(list_con_idx_y_preictal_remove.size):
        # Index of preictal period
        start_highlight_idx = y_preictal_remove[list_con_idx_y_preictal_remove[i]][0]
        end_highlight_idx = y_preictal_remove[list_con_idx_y_preictal_remove[i]][-1]

        if with_label:
            plt.axvspan(start_highlight_idx, end_highlight_idx + 1, ymax=1.15/1.2, color='k',
                        alpha=0.1, edgecolor='none',
                        label='Preictal (removed)')
            with_label = False
        else:
            plt.axvspan(start_highlight_idx, end_highlight_idx + 1, ymax=1.15/1.2, color='k',
                        alpha=0.1, edgecolor='none')

    # Highlight the withheld preictal periods
    with_label = True
    for i in range(list_con_idx_y_true.size):
        # Index of preictal period
        start_highlight_idx = y_true[list_con_idx_y_true[i]][0]
        end_highlight_idx = y_true[list_con_idx_y_true[i]][-1]

        if with_label:
            plt.axvspan(start_highlight_idx, end_highlight_idx + 1, ymax=1.15/1.2, color='orange',
                        alpha=0.3, edgecolor='none',
                        label='Preictal')
            with_label = False
        else:
            plt.axvspan(start_highlight_idx, end_highlight_idx + 1, ymax=1.15/1.2, color='orange',
                        alpha=0.3, edgecolor='none')

    # Highlight ictal periods and add ictal onset
    with_label = True
    for i in range(list_con_idx_y_ictal.size):
        # Index of ical period
        start_highlight_idx = y_ictal[list_con_idx_y_ictal[i]][0]
        end_highlight_idx = y_ictal[list_con_idx_y_ictal[i]][-1]

        if with_label:
            plt.axvspan(start_highlight_idx, end_highlight_idx + 1, ymax=1.15/1.2, color='r',
                        alpha=0.3, edgecolor='none',
                        label='Ictal')
            plt.axvline(start_highlight_idx, ymin=-0.2, ymax=1.15/1.2, color='k',
                        linestyle='-', linewidth=2.5,
                        marker='x', markersize=8, markeredgewidth=2.5,
                        label='Ictal onset')
            with_label = False
        else:
            plt.axvspan(start_highlight_idx, end_highlight_idx + 1, ymax=1.15/1.2, color='r',
                        alpha=0.3, edgecolor='none')
            plt.axvline(start_highlight_idx, ymin=-0.2, ymax=1.15/1.2, color='k',
                        linestyle='-', linewidth=2.5,
                        marker='x', markersize=8, markeredgewidth=2.5)

    # Alarm
    with_label = True
    for a in alarms:
        if with_label:
            plt.axvline(a, ymin=-0.2, ymax=1.15/1.2,
                        linestyle='-', color='r', linewidth=2.5,
                        marker='^', markeredgecolor='r', markersize=8, markeredgewidth=2.5,
                        label='Alarm')
            with_label = False
        # To avoid having multiple 'Alarm' legends
        else:
            plt.axvline(a, ymin=-0.2, ymax=1.15/1.2,
                        linestyle='-', color='r', linewidth=2.5,
                        marker='^', markeredgecolor='r', markersize=8, markeredgewidth=2.5)

    # Inhibited near alarm
    with_label = True
    for a in inhibited_alarms:
        if with_label:
            plt.axvline(a, ymin=-0.2, ymax=1.15/1.2,
                        linestyle='-', color='r', linewidth=2.5,
                        marker='^', markeredgecolor='r', markersize=8, markeredgewidth=2.5,
                        label='Alarm (inhibited)', alpha=0.3)
            with_label = False
        # To avoid having multiple 'Alarm' legends
        else:
            plt.axvline(a, ymin=-0.2, ymax=1.15/1.2,
                        linestyle='-', color='r', linewidth=2.5,
                        marker='^', markeredgecolor='r', markersize=8, markeredgewidth=2.5,
                        alpha=0.3)

    # Selected seizure alarm
    with_label = True
    for a in rm_select_sz_alarms:
        if with_label:
            plt.axvline(a, ymin=-0.2, ymax=1.15/1.2,
                        linestyle='-', color='c', linewidth=2.5,
                        marker='^', markeredgecolor='c', markersize=8, markeredgewidth=2.5,
                        label='Alarm (selected)', alpha=0.3)
            with_label = False
        # To avoid having multiple 'Alarm' legends
        else:
            plt.axvline(a, ymin=-0.2, ymax=1.15/1.2,
                        linestyle='-', color='c', linewidth=2.5,
                        marker='^', markeredgecolor='c', markersize=8, markeredgewidth=2.5,
                        alpha=0.3)

    # Removed seizure alarm
    with_label = True
    for a in rm_remove_sz_alarms:
        if with_label:
            plt.axvline(a, ymin=-0.2, ymax=1.15/1.2,
                        linestyle='-', color='k', linewidth=2.5,
                        marker='^', markeredgecolor='k', markersize=8, markeredgewidth=2.5,
                        label='Alarm (removed)', alpha=0.3)
            with_label = False
        # To avoid having multiple 'Alarm' legends
        else:
            plt.axvline(a, ymin=-0.2, ymax=1.15/1.2,
                        linestyle='-', color='k', linewidth=2.5,
                        marker='^', markeredgecolor='k', markersize=8, markeredgewidth=2.5,
                        alpha=0.3)

    plt.axhline(thd_firing_pow, color='green', linestyle='--', label='Threshold')
    plt.plot(range(y_hat.size), y_hat, 'k', alpha=0.4, label='SAE output')
    plt.plot(range(firing_pow.size), firing_pow, 'b', linewidth=2, label='FP output')
    leg = plt.legend(ncol=3)
    leg.get_frame().set_alpha(0.7)
    plt.ylim(0, 1.2)
    plt.yticks([0, thd_firing_pow, 1])
    plt.xlabel('Samples', fontsize=18)
    plt.ylabel('Firing Power (FP)', fontsize=18)
    plt.title('Threshold=' + str(thd_firing_pow) + ', Fold-CV=' + str(leave_out_seizure_idx_valid) +
              ', Fold-test=' + str(leave_out_seizure_idx_test), fontsize=20)
    plt.tight_layout()
    plt.xlim(xmin=0, xmax=y_hat.size)
    plt.savefig(os.path.join(figure_dir, 'prediction_thd_' + str(thd_firing_pow) +
                                         '_fold_' + str(leave_out_seizure_idx_valid) + '_' +
                                         str(leave_out_seizure_idx_test) +  '.png'),
                dpi=fig_save_dpi)
    ######################################################################

    plt.close('all')

    return result


def predict_loocv(patient_id, module_dir, data_dir,
                  list_thd_firing_pow, preictal_sec,
                  list_features, n_selected_features,
                  use_all_nonictals, use_available_preictal_period):

    # Get ictal (or seizure) information
    metadata_fname = os.path.join(os.path.join(data_dir, patient_id), 'trainset_' + str(preictal_sec) + '.mat')
    metadata_mat = loadmat(metadata_fname)

    n_seizures = metadata_mat.get('ictals').size
    segment_sec = metadata_mat.get('segment_sec')[0][0]
    n_extended_blocks_test = metadata_mat.get('n_extended_blocks_test')[0][0]
    n_hours_each_fold = 1.0 + (2 * n_extended_blocks_test)  # in hours

    print 'Use a model in ' + module_dir
    print 'Preictal seconds: ' + str(preictal_sec)
    print 'Use all nonictal data: ' + str(use_all_nonictals) + ' (this will not affect for ''valid'' and ''test'' sets)'
    print 'Use only available preictal period in the smooth the SAE output: ' + str(use_available_preictal_period)
    print 'List of threshold:', list_thd_firing_pow

    thd_results = []
    test_results = np.empty(list_thd_firing_pow.size, dtype=object)
    for thd_idx, thd_firing_pow in enumerate(list_thd_firing_pow):
        print ''
        print '-------------================ Threshold ' + str(thd_firing_pow) + ' ================-------------'

        t_results = []
        list_leave_out_seizure_idx_test = np.arange(n_seizures)
        valid_results = np.empty(list_leave_out_seizure_idx_test.size, dtype=object)
        for leave_out_seizure_idx_test in list_leave_out_seizure_idx_test:
            v_results = []
            # list_leave_out_seizure_idx_valid = np.setdiff1d(np.arange(n_seizures), leave_out_seizure_idx_test)
            list_leave_out_seizure_idx_valid = np.asarray([leave_out_seizure_idx_test])
            for leave_out_seizure_idx_valid in list_leave_out_seizure_idx_valid:
                save_model_dir = os.path.join(module_dir, patient_id + '/models_' + str(preictal_sec) + '_' +
                                                          str(leave_out_seizure_idx_valid) + '_' +
                                                          str(leave_out_seizure_idx_test))
                model_path = os.path.join(save_model_dir, 'sae.pkl')

                # Get data set for each fold
                dataset = EpilepsiaeFeatureLoaderSAE(patient_id=patient_id,
                                                     which_set='valid',
                                                     list_features=list_features,
                                                     leave_out_seizure_idx_valid=leave_out_seizure_idx_valid,
                                                     leave_out_seizure_idx_test=leave_out_seizure_idx_test,
                                                     data_dir=data_dir,
                                                     preictal_sec=preictal_sec,
                                                     use_all_nonictals=use_all_nonictals,
                                                     n_selected_features=n_selected_features,
                                                     preprocessor_dir=save_model_dir)

                # Load model
                model = serial.load(model_path)

                # Get prediction
                y_hat = predict(model, dataset)

                # Get ground truth
                y = np.argmax(dataset.y, axis=1)

                # Get all preictal labels (this might contain other preictal data due to the extended blocks contain seizures)
                preictal_labels = dataset.y_label_all

                # Get ictal (or seizure) labels
                ictal_labels = dataset.ictal_labels

                # Get prediction performance
                result = get_prediction_performance(y_hat=y_hat,
                                                    y=y,
                                                    preictal_labels=preictal_labels,
                                                    ictal_labels=ictal_labels,
                                                    leave_out_seizure_idx_valid=leave_out_seizure_idx_valid,
                                                    leave_out_seizure_idx_test=leave_out_seizure_idx_test,
                                                    figure_dir=module_dir + '/' + patient_id,
                                                    thd_firing_pow=thd_firing_pow,
                                                    preictal_sec=preictal_sec,
                                                    segment_sec=segment_sec,
                                                    use_available_preictal=use_available_preictal_period)

                result.loc[:,'sz_fname'] = pd.Series(metadata_mat['ictals'][leave_out_seizure_idx_valid][0]['filename'][0][0][0][0][0],
                                                     index=result.index)

                v_results.append(result)

            valid_results[leave_out_seizure_idx_test] = pd.concat(v_results)

            # Get the statistics of the results
            n_seizures = np.sum(valid_results[leave_out_seizure_idx_test]['n_seizures'].values)
            n_detected_sz = np.sum(valid_results[leave_out_seizure_idx_test]['detected_sz'].values)
            n_fp = np.sum(valid_results[leave_out_seizure_idx_test]['fp'].values)
            t_before_sz = np.sum(valid_results[leave_out_seizure_idx_test]['t_before_sz'][
                                     valid_results[leave_out_seizure_idx_test]['t_before_sz'] >= 0])
            sz_fname = metadata_mat['ictals'][leave_out_seizure_idx_test][0]['filename'][0][0][0][0][0]

            result = pd.DataFrame({
                'n_seizures': n_seizures,
                'n_detected_sz': n_detected_sz,
                'n_fp': n_fp,
                't_before_sz': t_before_sz,
                'n_hour': (n_seizures * n_hours_each_fold),
                'leave_out_sz': leave_out_seizure_idx_test,
                'sz_fname': sz_fname
            }, index=[0])

            t_results.append(result)

        test_results[thd_idx] = pd.concat(t_results)

        # Get the statistics of the results
        n_seizures = np.sum(test_results[thd_idx]['n_seizures'].values)
        n_detected_sz = np.sum(test_results[thd_idx]['n_detected_sz'].values)
        n_fp = np.sum(test_results[thd_idx]['n_fp'].values)
        t_before_sz = np.sum(test_results[thd_idx]['t_before_sz'][test_results[thd_idx]['t_before_sz'] >= 0])

        sensitivity = (n_detected_sz * 1.0) / n_seizures
        fpr = (n_fp * 1.0) / (n_seizures * n_hours_each_fold)
        if n_detected_sz > 0:
            avg_t_before_sz = (t_before_sz * 1.0) / n_detected_sz
        else:
            avg_t_before_sz = -1.0
            t_before_sz = -1.0

        result = pd.DataFrame({
            'n_seizures': n_seizures,
            'n_detected_sz': n_detected_sz,
            'n_fp': n_fp,
            't_before_sz': t_before_sz,
            'sensitivity': sensitivity,
            'fpr': fpr,
            'avg_t_before_sz': avg_t_before_sz,
            'n_hour': (n_seizures * n_hours_each_fold),
            'threshold': thd_firing_pow
        }, index=[0])

        thd_results.append(result)

    summary = pd.concat(thd_results)

    print ''
    print ''
    print '-------------================ Summary ================-------------'
    print ''
    for thd_idx, thd_firing_pow in enumerate(list_thd_firing_pow):
        print 'Result for threshold=' + str(thd_firing_pow) + ':'
        print test_results[thd_idx]
        print ''
    print ''
    print summary


def main():
    patient_id = 'pat_102'
    data_dir = params.DATA_DIR

    n_layers = 2
    module_dir = os.path.join(params.MODULE_DIR, 'sae_feature/sae_' + str(n_layers) + '_layer')

    n_selected_features = -1    # Don't use feature selection
    preictal_sec = 40 * 60
    use_all_nonictals = True
    use_available_preictal_period = True
    list_thd_firing_pow = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])

    predict_loocv(patient_id=patient_id,
                  module_dir=module_dir,
                  data_dir=data_dir,
                  list_thd_firing_pow=list_thd_firing_pow,
                  preictal_sec=preictal_sec,
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
                  n_selected_features=n_selected_features,
                  use_all_nonictals=use_all_nonictals,
                  use_available_preictal_period=use_available_preictal_period)


if __name__ == '__main__':
    main()
