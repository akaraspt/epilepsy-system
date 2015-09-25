import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from sklearn import preprocessing

from pylearn2ext.epilepsiae import EpilepsiaeDatasetExtractor


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


def plot_eeg_seizure(X, channel_labels, seizure_seconds, sampling_rate,
                     interval_seconds=100, min_X=-1024, max_X=1024, is_scale=True,
                     figure_width=800, figure_height=600, figure_output_path='figure'):
    # Metadata of the file
    n_channels = X.shape[0]
    n_data = X.shape[1]

    # Butterworth filter
    f_nyq = sampling_rate * 0.5
    Wlow = 3 / f_nyq
    Whigh = 30 / f_nyq
    (b, a) = butter(4, [Wlow, Whigh], btype='bandpass')
    data = filtfilt(b, a, X, axis=1)

    # Preprocessing
    if is_scale:
        scaler = preprocessing.StandardScaler()
        scaler = scaler.fit(data.transpose())
        data = scaler.transform(data.transpose()).transpose()

    # Seizure periods
    idx_end_seizure_sample = np.where(np.diff(seizure_seconds) > 1)[0]
    if idx_end_seizure_sample.size == 0:
        idx_seizure_sample = [np.arange(seizure_seconds.size)]
    else:
        idx_seizure_sample = gen_idx_range(idx_end_seizure_sample, seizure_seconds.size)

    # Display data
    intervals_sample = interval_seconds * sampling_rate
    n_intervals = int(math.ceil(n_data / (intervals_sample * 1.0)))
    for i in range(n_intervals):
        start_int_idx = i * intervals_sample
        end_int_idx = (i+1) * intervals_sample

        start_time = start_int_idx / sampling_rate
        end_time = end_int_idx / sampling_rate

        print('Plot figure for interval: ' + str(start_time) + ':' + str(end_time))

        if end_int_idx > n_data:
            end_int_idx = n_data

        display_data = data[:,start_int_idx:end_int_idx]
        n_display_data = display_data.shape[1]

        # Time
        t = (interval_seconds * (np.arange(n_display_data, dtype=float) / n_display_data)) + (i*interval_seconds)

        # Figure properties
        fig_dpi = 32
        fig_width = figure_width/fig_dpi
        fig_height = figure_height/fig_dpi
        plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)

        # Adjust axes
        ax = plt.gca()
        plt.xlim([start_time, end_time])
        plt.xticks(np.arange(start=start_time, stop=end_time, step=interval_seconds/10), fontsize=40)
        if is_scale:
            dmin = -5
            dmax = 5
        else:
            dmin = min_X
            dmax = max_X
        dr = (dmax - dmin) * 0.95 # Crowd them a bit.
        y0 = dmin
        y1 = ((n_channels-1) * dr) + dmax
        plt.ylim(y0, y1)

        # Plot EEG from multiple channels
        segs = []
        ticklocs = []
        offsets = np.zeros((n_channels,2), dtype=float)
        for c in range(n_channels):
            segs.append(np.hstack((t[:,np.newaxis], display_data[c,:,np.newaxis])))
            ticklocs.append(c*dr)
        offsets[:,1] = ticklocs
        lines = LineCollection(segs,
                               offsets=offsets,
                               transOffset=None)
        ax.add_collection(lines)
        ax.set_yticks(ticklocs)
        ax.set_yticklabels(channel_labels, fontsize=30)

        # Highlight seizure period
        for idx in idx_seizure_sample:
            seizure_interval = np.intersect1d(seizure_seconds[idx], np.arange(start=start_time, stop=end_time))
            if seizure_interval.size > 0:
                if seizure_seconds[idx][0] in seizure_interval:
                    plt.axvline(seizure_interval[0], color='red', linewidth=4, linestyle='-')

                if seizure_seconds[idx][-1] in seizure_interval:
                    plt.axvline(seizure_interval[-1], color='red', linewidth=4, linestyle='-')
                    # plt.axvspan(idx[0], idx[-1], color='red', alpha=0.3)
                # else:
                    # plt.axvspan(idx[0], end_time, color='red', alpha=0.3)

        plt.xlabel('Time (s)', fontsize=40)
        plt.tight_layout()
        plt.savefig(figure_output_path + '/' + str(i) + '.png')


def plot_eeg_predict_seizure(X, y, channel_labels, seizure_seconds, sampling_rate,
                             interval_seconds=100, min_X=-1024, max_X=1024, is_scale=True,
                             figure_width=800, figure_height=600, figure_output_path='figure'):
    # Metadata of the file
    n_channels = X.shape[0]
    n_data = X.shape[1]

    # # Butterworth filter
    # f_nyq = sampling_rate * 0.5
    # Wlow = 3 / f_nyq
    # Whigh = 30 / f_nyq
    # (b, a) = butter(4, [Wlow, Whigh], btype='bandpass')
    # data = filtfilt(b, a, X, axis=1)

    data = X
    predict = y

    # Preprocessing
    if is_scale:
        scaler = preprocessing.StandardScaler()
        scaler = scaler.fit(data.transpose())
        data = scaler.transform(data.transpose()).transpose()

    # Seizure periods
    idx_end_seizure_sample = np.where(np.diff(seizure_seconds) > 1)[0]
    if idx_end_seizure_sample.size == 0:
        idx_seizure_sample = [np.arange(seizure_seconds.size)]
    else:
        idx_seizure_sample = gen_idx_range(idx_end_seizure_sample, seizure_seconds.size)

    # Display data
    intervals_sample = interval_seconds * sampling_rate
    n_intervals = int(math.ceil(n_data / (intervals_sample * 1.0)))
    for i in range(n_intervals):
        start_int_idx = i * intervals_sample
        end_int_idx = (i+1) * intervals_sample

        start_time = start_int_idx / sampling_rate
        end_time = end_int_idx / sampling_rate

        print('Plot figure for interval: ' + str(start_time) + ':' + str(end_time))

        if end_int_idx > n_data:
            end_int_idx = n_data

        display_data = data[:,start_int_idx:end_int_idx]
        display_predict = predict[start_int_idx:end_int_idx]
        n_display_data = display_data.shape[1]

        # Time
        t = (interval_seconds * (np.arange(n_display_data, dtype=float) / n_display_data)) + (i*interval_seconds)

        # Figure properties
        fig_dpi = 32
        fig_width = figure_width/fig_dpi
        fig_height = figure_height/fig_dpi
        plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)

        # Adjust axes
        ax = plt.gca()
        plt.xlim([start_time, end_time])
        plt.xticks(np.arange(start=start_time, stop=end_time, step=interval_seconds/10), fontsize=25)
        if is_scale:
            dmin = -5
            dmax = 5
        else:
            dmin = min_X
            dmax = max_X
        dr = (dmax - dmin) * 0.95 # Crowd them a bit.
        y0 = dmin
        y1 = ((n_channels-1) * dr) + dmax
        plt.ylim(y0, y1)

        # Plot EEG from multiple channels
        segs = []
        ticklocs = []
        offsets = np.zeros((n_channels,2), dtype=float)
        for c in range(n_channels):
            segs.append(np.hstack((t[:,np.newaxis], display_data[c,:,np.newaxis])))
            ticklocs.append(c*dr)
        offsets[:,1] = ticklocs
        lines = LineCollection(segs,
                               offsets=offsets,
                               transOffset=None)
        ax.add_collection(lines)
        ax.set_yticks(ticklocs)
        ax.set_yticklabels(channel_labels, fontsize=25)

        # Highlight seizure period
        for idx in idx_seizure_sample:
            seizure_interval = np.intersect1d(seizure_seconds[idx], np.arange(start=start_time, stop=end_time))
            if seizure_interval.size > 0:
                if seizure_seconds[idx][0] in seizure_interval:
                    plt.axvline(seizure_interval[0], color='red', linewidth=4, linestyle='-')

                if seizure_seconds[idx][-1] in seizure_interval:
                    plt.axvline(seizure_interval[-1], color='red', linewidth=4, linestyle='-')
                    # plt.axvspan(idx[0], idx[-1], color='red', alpha=0.3)
                # else:
                    # plt.axvspan(idx[0], end_time, color='red', alpha=0.3)

        plt.xlabel('Time (s)', fontsize=30)
        plt.ylabel('Scalp EEG Channels', fontsize=30)
        plt.tight_layout()
        plt.savefig(figure_output_path + '/' + str(i) + '_eeg.png')

        # Figure properties
        fig_dpi = 32
        fig_width = figure_width/fig_dpi
        fig_height = (figure_height/3)/fig_dpi
        plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)

        plt.plot(t, display_predict, linewidth=4)
        ax = plt.gca()
        ax.yaxis.grid(True)
        plt.ylim([0, 20])
        plt.yticks(np.arange(start=5, stop=20, step=5), fontsize=25)
        plt.xlim([start_time, end_time])
        plt.xticks(np.arange(start=start_time, stop=end_time, step=interval_seconds/10), fontsize=25)
        plt.ylabel('# of Seizure Ch.', fontsize=30)
        plt.xlabel('Time (s)', fontsize=30)
        plt.tight_layout()
        plt.savefig(figure_output_path + '/' + str(i) + '_predict.png')


def plot_eeg_predict_seizure_period(X, y, channel_labels, seizure_seconds, sampling_rate,
                                    start_second, end_second, min_X=-1024, max_X=1024, is_scale=True,
                                    n_X_ticks=10, channel_th_y_lim=None,
                                    figure_width=800, figure_height=600, figure_output_path='figure'):
    # Metadata of the file
    n_channels = X.shape[0]
    n_data = X.shape[1]

    # # Butterworth filter
    # f_nyq = sampling_rate * 0.5
    # Wlow = 3 / f_nyq
    # Whigh = 30 / f_nyq
    # (b, a) = butter(4, [Wlow, Whigh], btype='bandpass')
    # data = filtfilt(b, a, X, axis=1)

    data = X
    predict = y

    # Preprocessing
    if is_scale:
        scaler = preprocessing.StandardScaler()
        scaler = scaler.fit(data.transpose())
        data = scaler.transform(data.transpose()).transpose()

    # Seizure periods
    idx_end_seizure_sample = np.where(np.diff(seizure_seconds) > 1)[0]
    if idx_end_seizure_sample.size == 0:
        idx_seizure_sample = [np.arange(seizure_seconds.size)]
    else:
        idx_seizure_sample = gen_idx_range(idx_end_seizure_sample, seizure_seconds.size)

    # Display data

    start_int_idx = (start_second-1) * sampling_rate
    end_int_idx = end_second * sampling_rate

    start_time = start_second
    end_time = end_second

    interval_seconds = end_time - start_time + 1

    print('Plot figure for interval: ' + str(start_time) + ':' + str(end_time))

    if end_int_idx > n_data:
        end_int_idx = n_data

    display_data = data[:,start_int_idx:end_int_idx]
    display_predict = predict[start_int_idx:end_int_idx]
    n_display_data = display_data.shape[1]

    # Time
    t = (interval_seconds * (np.arange(n_display_data, dtype=float) / n_display_data)) + start_second

    # Figure properties
    fig_dpi = 32
    fig_width = figure_width/fig_dpi
    fig_height = figure_height/fig_dpi
    plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)

    # Adjust axes
    ax = plt.gca()
    plt.xlim([start_time, end_time])
    plt.xticks(np.arange(start=start_time, stop=end_time, step=interval_seconds/n_X_ticks), fontsize=25)
    if is_scale:
        dmin = -5
        dmax = 5
    else:
        dmin = min_X
        dmax = max_X
    dr = (dmax - dmin) * 0.95 # Crowd them a bit.
    y0 = dmin
    y1 = ((n_channels-1) * dr) + dmax
    plt.ylim(y0, y1)

    # Add number of channel labels
    channel_labels_nums = [channel_labels[i] + ' (' + str(i+1) + ')' for i in range(channel_labels.size)]

    # Plot EEG from multiple channels
    segs = []
    ticklocs = []
    offsets = np.zeros((n_channels,2), dtype=float)
    for c in range(n_channels):
        segs.append(np.hstack((t[:,np.newaxis], display_data[c,:,np.newaxis])))
        ticklocs.append(c*dr)
    offsets[:,1] = ticklocs
    lines = LineCollection(segs,
                           offsets=offsets,
                           transOffset=None)
    ax.add_collection(lines)
    ax.set_yticks(ticklocs)
    ax.set_yticklabels(channel_labels_nums, fontsize=25)

    # Highlight seizure period
    for idx in idx_seizure_sample:
        seizure_interval = np.intersect1d(seizure_seconds[idx], np.arange(start=start_time, stop=end_time))
        if seizure_interval.size > 0:
            if seizure_seconds[idx][0] in seizure_interval:
                plt.axvline(seizure_interval[0], color='red', linewidth=4, linestyle='-')

            if seizure_seconds[idx][-1] in seizure_interval:
                plt.axvline(seizure_interval[-1], color='red', linewidth=4, linestyle='-')
                # plt.axvspan(idx[0], idx[-1], color='red', alpha=0.3)
            # else:
                # plt.axvspan(idx[0], end_time, color='red', alpha=0.3)

    plt.xlabel('Time (s)', fontsize=30)
    plt.ylabel('Scalp EEG Channels', fontsize=30)
    plt.tight_layout()
    plt.savefig(figure_output_path + '/' + 'eeg.png')

    # Figure properties
    fig_dpi = 32
    fig_width = figure_width/fig_dpi
    fig_height = (figure_height/3)/fig_dpi
    plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)

    plt.plot(t, display_predict, linewidth=4)
    ax = plt.gca()
    ax.yaxis.grid(True)
    plt.ylim(channel_th_y_lim)
    plt.yticks(np.arange(start=5, stop=20, step=5), fontsize=25)
    plt.xlim([start_time, end_time])
    plt.xticks(np.arange(start=start_time, stop=end_time, step=interval_seconds/10), fontsize=25)
    plt.ylabel('# of Seizure Ch.', fontsize=30)
    plt.xlabel('Time (s)', fontsize=30)
    plt.tight_layout()
    plt.savefig(figure_output_path + '/' + 'predict.png')


def plot_chbmit_eeg(file_path):
    # Load data from MATLAB file
    mat = loadmat(file_path)
    X = mat['X']
    channel_labels = np.asarray([l[0][0] for l in mat['labels']])
    seizure_seconds = mat['seizure_second'][0]
    sampling_rate = mat['sampling_rate'][0, 0]
    plot_eeg_seizure(X=X,
                     channel_labels=channel_labels,
                     seizure_seconds=seizure_seconds,
                     sampling_rate=sampling_rate,
                     interval_seconds=100,
                     is_scale=True,
                     figure_width=1500,
                     figure_height=1200,
                     figure_output_path='../figure')


def plot_epilepsiae_eeg(data_dir, filename, metafile):
    # Get seizure information
    seizure_info = pd.read_table(os.path.join(data_dir, metafile), sep='\t')
    seizure_info['filename'] = seizure_info['filename'].str.replace('.data', '.mat', case=False)

    data_ext = EpilepsiaeDatasetExtractor(patient_id=1, files=filename,
                                          seizure_info=seizure_info,
                                          data_dir=data_dir)
    X, raw_labels, channel_labels, \
    seizure_range_idx, seizure_range_second, seizure_seconds, \
    n_channels, sample_size, sampling_rate = data_ext.load_source_data(sample_size_second=1)

    plot_eeg_seizure(X=X,
                     channel_labels=channel_labels,
                     seizure_seconds=seizure_seconds,
                     sampling_rate=sampling_rate,
                     interval_seconds=100,
                     is_scale=True,
                     figure_width=1500,
                     figure_height=1200,
                     figure_output_path='../figure')


if __name__ == '__main__':
    plot_chbmit_eeg(file_path=os.path.join('/Users/akara/Workspace/data/chbmit',
                                           'chb20/chb20_12_mod.mat'))

    # plot_epilepsiae_eeg(data_dir='/Users/akara/Workspace/data/epilepsiae',
    #                     filename=['rec_26402102/26402102_0003.mat'],
    #                     metafile='RECORDS-WITH-SEIZURES.txt')


# # Load data from MATLAB file
# mat = loadmat(os.path.join(data_dir, filename))
# X = mat['signals']
# channel_labels = np.asarray(mat['elec_names'][0][1:-1].split(','))
# sampling_rate = mat['sample_freq'][0, 0]
#
# # Get seizure information
# seizure_files = pd.read_table(os.path.join(data_dir, metafile), sep='\t')
# seizure_files['filename'] = seizure_files['filename'].str.replace('.data', '.mat', case=False)
#
# start_ts = datetime.datetime.strptime(mat['start_ts'][0], '%Y-%m-%d %H:%M:%S.%f')
# match_files = seizure_files[seizure_files['filename'].str.contains(filename)]
# seizure_seconds = np.empty(0, dtype=int)
# if match_files.shape[0] > 0:
#     for index, row in match_files.iterrows():
#         start_seizure_idx = row['onset_sample']
#         end_seizure_idx = row['offset_sample']
#
#         print 'Seizure ts:', row['onset'], row['offset']
#         print 'Seizure samples:', start_seizure_idx, end_seizure_idx
#
#         # Store index of seizure seconds
#         start_seizure_second_idx = (start_seizure_idx) / (sampling_rate * 1.0)
#         end_seizure_second_idx = (end_seizure_idx) / (sampling_rate * 1.0)
#         _temp_seizure_seconds = np.arange(int(math.floor(start_seizure_second_idx)),
#                                           int(math.ceil(end_seizure_second_idx)) + 1) # Plus 1 due to the nature of np.arange
#         seizure_seconds = np.append(seizure_seconds, _temp_seizure_seconds)
#
#         start_seizure_ts = start_ts + datetime.timedelta(seconds=_temp_seizure_seconds[0])
#         end_seizure_ts = start_ts + datetime.timedelta(seconds=_temp_seizure_seconds[-1])
#         print 'Figure start seizure:', start_seizure_ts
#         print 'Figure end seizure:', end_seizure_ts