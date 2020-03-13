import numpy as np
import pandas as pd
import neurokit as nk
from scipy.signal import find_peaks
from scipy.stats import linregress
# pip install neurokit
# pip install pathlib

ND = 0
BASELINE_STATE = 1
STRESS_STATE = 2
AMUSEMENT_STATE = 3
NONE1 = 4
NONE1 = 5
NONE1 = 6
NONE1 = 7


def print_obj_len(obj, id=''):
    print ('=== ' + id + ' ===')
    dict_length = {key: len(value) for key, value in obj.items()}
    print(dict_length)


def get_slope(series):
    print(len(np.arange(len(series))))
    print(len(series))
    linreg = linregress(np.arange(len(series)), series )
    slope = linreg[0]
    return slope


def extract_default_features(signal):
    """
    Function to compute default features that the most of signals need
    :param signal: Signal to extract the features
    :return: mean, std, min, max, dynamic range
    """
    mx = np.amax(signal)
    mn = np.amin(signal)
    return np.mean(signal), np.std(signal), mn, mx, (mx - mn)


def extract_mean_std_features(signal_data, block=700, shift=700, pref=''):
    mean_features = []
    std_features = []
    max_features = []
    min_features = []

    i = 0
    while i < len(signal_data):
        temp = signal_data[i:i + block]
        mean_features.append(np.mean(temp))
        std_features.append(np.std(temp))
        min_features.append(np.amin(temp))
        max_features.append(np.amax(temp))
        i += shift
    features = {pref+'mean':mean_features, pref+'std':std_features, pref+'min':min_features, pref+'max':max_features}
    #one_set = np.column_stack((mean_features, std_features, min_features, max_features))
    #return one_set
    return features


def get_acc_features(signal, sample_size, window_size=1, shift=1):
    i = 0
    features = {
        'x_min': [], 'x_max': [], 'x_mean': [], 'x_std': [], 'x_absint': [], 'x_peakf': [],
        'y_min': [], 'y_max': [], 'y_mean': [], 'y_std': [], 'y_absint': [], 'y_peakf': [],
        'z_min': [], 'z_max': [], 'z_mean': [], 'z_std': [], 'z_absint': [], 'z_peakf': [],
        'a_mean': [], 'a_std': [], 'a_absint': []
    }
    signal = np.array(signal)
    xdata = signal[:, 0]
    ydata = signal[:, 1]
    zdata = signal[:, 2]
    window_size = int(sample_size * window_size)
    shift = int(sample_size * shift)
    while i < len(signal):
        temp = xdata[i: i + window_size]
        m, s, mn, mx, _ = extract_default_features(temp)
        features['x_min'].append(mn)
        features['x_max'].append(mx)
        features['x_mean'].append(m)
        features['x_std'].append(s)
        features['x_absint'].append(abs(np.trapz(temp)))
        peaks, properties = find_peaks(temp, height = 0)
        #mn_scr = np.mean(properties['peak_heights'])
        #std_scr = np.std(properties['peak_heights'])   
        #num_scr = np.size(peaks)
        #sum_scr = np.sum(properties['peak_heights'])
        #indexes = find_peaks(temp)# peakutils.indexes(temp)
        features['x_peakf'].append(np.size(peaks))

        temp = ydata[i: i + window_size]
        m, s, mn, mx, _ = extract_default_features(temp)
        features['y_min'].append(mn)
        features['y_max'].append(mx)
        features['y_mean'].append(m)
        features['y_std'].append(s)
        features['y_absint'].append(abs(np.trapz(temp)))
        peaks, properties = find_peaks(temp, height = 0)
        features['y_peakf'].append(np.size(peaks))

        temp = zdata[i: i + window_size]
        m, s, mn, mx, _ = extract_default_features(temp)
        features['z_min'].append(mn)
        features['z_max'].append(mx)
        features['z_mean'].append(m)
        features['z_std'].append(s)
        features['z_absint'].append(abs(np.trapz(temp)))
        peaks, properties = find_peaks(temp, height = 0)
        features['z_peakf'].append(np.size(peaks))

        features['a_mean'].append(features['x_mean'][-1] + features['y_mean'][-1] + features['z_mean'][-1])
        features['a_std'].append(features['x_std'][-1] + features['y_std'][-1] + features['z_std'][-1])
        features['a_absint'].append(features['x_absint'][-1] + features['y_absint'][-1] + features['z_absint'][-1])

        i += shift

    #ids = ['ACC_' + val for val, arr in features.items()]
    return features
    

# ECG and BVP
def get_ecg_features(signal, sample_size, window_size=1, shift=1):
    #print('ECG size: ' + str(sample_size))
    #print(len(signal) / sample_size)
    #processed = nk.ecg_process(ecg=signal, sampling_rate=sample_size, filter_frequency=sample_size/2-1)
    #print (processed)
    features = {}
    features = extract_mean_std_features(signal, int(sample_size*window_size), int(sample_size* shift))
    return features


def get_eda_features(signal, sample_size, window_size=1, shift=1):
    i = 0
    # primero aplicar un 5Hz lowpass filter
    features = {
        'min': [], 'max': [], 'mean': [], 'std': [], 'range': []#, 'slope': [],
        #'scr_mean': [], 'scr_std':[], 'scl_mean': [], 'scl_std':[],
        #'scl_corr': [], 'scr_segs': [],
        #'scr_sum_amp': [], 'scr_sum_t': [], 'scr_int': []
    }
    window_size = int(sample_size * window_size)
    shift = int(sample_size * shift)
    while i < len(signal):
        temp = signal[i: i + window_size]
        m, s, mn, mx, dr = extract_default_features(temp)
        features['min'].append(mn)
        features['max'].append(mx)
        features['mean'].append(m)
        features['std'].append(s)
        features['range'].append(dr)
        #features['slope'].append(get_slope(temp.dropna()))
        
        i += shift

    return features


def get_emg_features(signal, sample_size, window_size=1, shift=1):
    i = 0
    features = {
        'min': [], 'max': [], 'mean': [], 'std': [], 'range': [],
        'absint': [], 'median': [],
        #'p10th':[], 'p90th': [],
        'f_mean': [], 'f_median': [], 'npeaks': []#, 'peakf': [],
        #'psd': [], 'amp_mean': [], 'amp_std': [],
        #'amp_sum': [], 'amp_sum_norm': []
    }
    window_size = int(sample_size * window_size)
    shift = int(sample_size * shift)
    while i < len(signal):
        temp = signal[i: i + window_size]
        m, s, mn, mx, dr = extract_default_features(temp)
        features['min'].append(mn)
        features['max'].append(mx)
        features['mean'].append(m)
        features['std'].append(s)
        features['range'].append(dr)
        features['absint'].append(abs(np.trapz(temp)))
        features['median'].append(np.median(temp))
        peaks, properties = find_peaks(temp, height = 0)
        features['npeaks'].append(np.size(peaks))
        features['f_mean'].append(np.mean(properties['peak_heights']))
        features['f_median'].append(np.median(properties['peak_heights']))
        
        i += shift

    return features


def get_resp_features(signal, sample_size, window_size=1, shift=1):
    features = {}
    features = extract_mean_std_features(signal, int(sample_size*window_size), int(sample_size* shift))
    return features


def get_temp_features(signal, sample_size, window_size=1, shift=1):
    i = 0
    features = {
        'min': [], 'max': [], 'mean': [], 'std': [], 'range': []#, 'slope': []
    }
    window_size = int(sample_size * window_size)
    shift = int(sample_size * shift)
    while i < len(signal):
        temp = signal[i: i + window_size]
        m, s, mn, mx, dr = extract_default_features(temp)
        features['min'].append(mn)
        features['max'].append(mx)
        features['mean'].append(m)
        features['std'].append(s)
        features['range'].append(dr)
        #features['slope'].append(get_slope(temp))
        
        i += shift

    return features


def get_final_labels(labels, sample_size=700, window_size=1, shift=1, binary=False):
    i = 0
    output = []
    window_size = int(sample_size * window_size)
    shift = int(sample_size * shift)
    while i < len(labels):
        temp = labels[i: i + window_size]
        ltemp = [np.count_nonzero(temp == state) for state in list(range(8))]
        idx = np.argmax(ltemp)
        if(binary):
            if((ltemp[BASELINE_STATE] + ltemp[AMUSEMENT_STATE]) >= ltemp[idx]):
                idx = BASELINE_STATE
        output.append(idx)
        
        i += shift
    return output