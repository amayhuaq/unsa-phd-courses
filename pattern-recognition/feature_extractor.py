import numpy as np
import pandas as pd
import neurokit as nk
from scipy.signal import find_peaks
from scipy.stats import linregress, pearsonr
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
NO_PEAK = -1
NONE_VAL = -9999


def print_obj_len(obj, id=''):
    print ('=== ' + id + ' ===')
    dict_length = {key: len(value) for key, value in obj.items()}
    print(dict_length)


def get_slope(series):
    if len(series) == 0:
        return NONE_VAL
    linreg = linregress(np.arange(len(series)), series )
    slope = linreg[0]
    return slope

def get_peak_frequency(signal, ini=0):
    peaks, properties = find_peaks(signal, height = 0)
    if len(peaks) == 0:
        return NO_PEAK
    else:
        pos = np.argmax(properties['peak_heights'])
        return ini + peaks[pos]



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
    return features


def get_acc_features(signal, sample_size, window_size=1, shift=1):
    i = 0
    features = {
        'x_mean': [], 'x_std': [], 'x_absint': [], 'x_peakf': [],
        'y_mean': [], 'y_std': [], 'y_absint': [], 'y_peakf': [],
        'z_mean': [], 'z_std': [], 'z_absint': [], 'z_peakf': [],
        'a_mean': [], 'a_std': [], 'a_absint': []
    }
    signal = np.array(signal)
    xdata = signal[:, 0]
    ydata = signal[:, 1]
    zdata = signal[:, 2]
    data3d = np.sqrt(xdata**2 + ydata**2 + zdata**2)
    window_size = int(sample_size * window_size)
    shift = int(sample_size * shift)
    while i < len(signal):
        temp = xdata[i: i + window_size]
        m, s, mn, mx, _ = extract_default_features(temp)
        features['x_mean'].append(m)
        features['x_std'].append(s)
        features['x_absint'].append(abs(np.trapz(temp)))
        features['x_peakf'].append(get_peak_frequency(temp))

        temp = ydata[i: i + window_size]
        m, s, mn, mx, _ = extract_default_features(temp)
        features['y_mean'].append(m)
        features['y_std'].append(s)
        features['y_absint'].append(abs(np.trapz(temp)))
        features['y_peakf'].append(get_peak_frequency(temp))

        temp = zdata[i: i + window_size]
        m, s, mn, mx, _ = extract_default_features(temp)
        features['z_mean'].append(m)
        features['z_std'].append(s)
        features['z_absint'].append(abs(np.trapz(temp)))
        features['z_peakf'].append(get_peak_frequency(temp))
        
        temp = data3d[i: i + window_size]
        m, s, mn, mx, _ = extract_default_features(temp)
        features['a_mean'].append(m)
        features['a_std'].append(s)
        features['a_absint'].append(abs(np.trapz(temp)))
        
        i += shift

    return features
    

# ECG and BVP
def get_ecg_features(signal, sample_size, window_size=1, shift=1):
    i = 0
    features = {
        'hr_mean': [], 'hr_std': [], 
        'sum_ulf': [], 'sum_lf':[], 'sum_hf':[], 'sum_uhf':[]#,
        #'hrv_mean': [], 'hrv_std': [],
        #'pNN50': [], 'TINN': [], 'rms': [], 'LFHF': [],
        #'LFn': [], 'HFn': [], 
        #'rel': [], 'e_ulf': [], 'e_lf':[], 'e_hf':[], 'e_uhf':[]
    }
    signal = np.array(signal).flatten()
    s_processed = nk.ecg_process(ecg=signal, sampling_rate=sample_size, filter_type=None)
    # Index(['ECG_Raw', 'ECG_Filtered', 'ECG_R_Peaks', 'Heart_Rate', 'ECG_Systole', 
    #   'ECG_Signal_Quality', 'ECG_RR_Interval', 'ECG_HRV_ULF', 'ECG_HRV_VLF',
    #   'ECG_HRV_LF', 'ECG_HRV_HF', 'ECG_HRV_VHF'],
    #  dtype='object')
    hr = s_processed['df']['Heart_Rate']
    #print(s_processed['ECG']['HRV'].keys())
    #print(s_processed['ECG']['HRV']['RR_Intervals'])
    #hrv = s_processed['ECG']['HRV']['RR_Intervals']
    
    window_size = int(sample_size * window_size)
    shift = int(sample_size * shift)
    
    while i < len(signal):
        temp = hr[i: i + window_size]
        m, s, mn, mx, dr = extract_default_features(temp)
        features['hr_mean'].append(m)
        features['hr_std'].append(s)
        
        temp = s_processed['df']['ECG_HRV_VLF'][i: i + window_size]
        features['sum_ulf'].append(np.sum(temp))
        temp = s_processed['df']['ECG_HRV_LF'][i: i + window_size]
        features['sum_lf'].append(np.sum(temp))
        temp = s_processed['df']['ECG_HRV_HF'][i: i + window_size]
        features['sum_hf'].append(np.sum(temp))
        temp = s_processed['df']['ECG_HRV_VHF'][i: i + window_size]
        features['sum_uhf'].append(np.sum(temp))
        
        """
        features['hrv_mean'].append(hrv['meanNN'])
        features['hrv_std'].append(hrv['sdNN'])
        features['pNN50'].append(hrv['pNN50'])
        features['TINN'].append(hrv['Triang'])
        features['rms'].append(hrv['RMSSD'])
        features['LFHF'].append(hrv['LF/HF'])
        features['LFn'].append(hrv['LFn'])
        features['HFn'].append(hrv['HFn'])
        features['rel'].append(hrv['Total_Power'])
        features['e_ulf'].append(hrv['ULF'])
        features['e_lf'].append(hrv['LF'])
        features['e_hf'].append(hrv['HF'])
        features['e_uhf'].append(hrv['VHF'])
        """
        
        i += shift

    return features


def get_eda_features(signal, sample_size, window_size=1, shift=1):
    i = 0
    features = {
        'min': [], 'max': [], 'mean': [], 'std': [], 'range': [], 'slope': [],
        'scl_mean': [], 'scl_std':[], 'scr_std':[],
        'scl_corr': [], 'scr_segs': [],
        'scr_sum_amp': [], 'scr_sum_t': [], 'scr_area': []
    }
    signal = np.array(signal).flatten()
    s_processed = nk.eda_process(eda=signal, sampling_rate=sample_size)
    signal = s_processed['df']["EDA_Filtered"]
    scl = s_processed['df']["EDA_Tonic"]
    scr = s_processed['df']["EDA_Phasic"]
        
    window_size = int(sample_size * window_size)
    shift = int(sample_size * shift)
    duration = s_processed['EDA']['SCR_Peaks_Indexes'] - s_processed['EDA']['SCR_Onsets']
    ampl = s_processed['EDA']['SCR_Peaks_Amplitudes']
    
    while i < len(signal):
        temp = signal[i: i + window_size]
        m, s, mn, mx, dr = extract_default_features(temp)
        features['min'].append(mn)
        features['max'].append(mx)
        features['mean'].append(m)
        features['std'].append(s)
        features['range'].append(dr)
        features['slope'].append(get_slope(temp))
        
        temp = scl[i: i + window_size]
        m, s, mn, mx, dr = extract_default_features(temp)
        features['scl_mean'].append(m)
        features['scl_std'].append(s)
        time = [k * 1.0/sample_size for k in np.arange(len(temp))]
        corr, _ = pearsonr(time, temp)
        features['scl_corr'].append(corr)
        
        temp = scr[i: i + window_size]
        m, s, mn, mx, dr = extract_default_features(temp)
        features['scr_std'].append(s)
        
        temp = s_processed['EDA']['SCR_Peaks_Indexes']
        sel = (temp >= i) & (temp < (i + window_size))
        features['scr_segs'].append(np.count_nonzero(sel))
        features['scr_sum_amp'].append(np.sum(ampl[sel]))
        features['scr_sum_t'].append(np.sum(duration[sel]))
        features['scr_area'].append(np.sum(0.5 * ampl[sel] * duration[sel]))
        
        i += shift

    return features


def get_emg_features(signal, sample_size, window_size=1, shift=1):
    i = 0
    features = {
        'min': [], 'max': [], 'mean': [], 'std': [], 'range': [],
        'absint': [], 'median': [],
        #'p10th':[], 'p90th': [],
        #'f_mean': [], 'f_median': [], 
        'npeaks': [], 'peakf': [],
        #'psd': [], 
        'amp_mean': [], 'amp_std': [],
        'amp_sum': []#, 'amp_sum_norm': []
    }
    window_size = int(sample_size * window_size)
    shift = int(sample_size * shift)
    signal = np.array(signal).flatten()
    
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
        features['peakf'].append(get_peak_frequency(temp))
        peaks, properties = find_peaks(temp, height = 0)
        features['npeaks'].append(np.size(peaks))
        features['amp_mean'].append(np.mean(properties['peak_heights']))
        features['amp_std'].append(np.std(properties['peak_heights']))
        features['amp_sum'].append(np.sum(properties['peak_heights']))
        
        i += shift

    return features


def get_resp_features(signal, sample_size, window_size=1, shift=1):
    features = {}
    features = extract_mean_std_features(signal, int(sample_size*window_size), int(sample_size* shift))
    return features


def get_temp_features(signal, sample_size, window_size=1, shift=1):
    i = 0
    features = {
        'min': [], 'max': [], 'mean': [], 'std': [], 'range': [], 'slope': []
    }
    signal = np.array(signal).flatten()
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
        features['slope'].append(get_slope(temp))
        
        i += shift

    return features


def get_final_labels(labels, sample_size=700, window_size=1, shift=1, binary=False):
    i = 0
    output = []
    window_size = int(sample_size * window_size)
    shift = int(sample_size * shift)
    while i < len(labels):
        temp = labels[i: i + window_size]
        ltemp = [np.count_nonzero(temp == state) for state in np.arange(8)]
        idx = np.argmax(ltemp)
        if(binary):
            if((ltemp[BASELINE_STATE] + ltemp[AMUSEMENT_STATE]) >= ltemp[idx]):
                idx = BASELINE_STATE
        output.append(idx)
        
        i += shift
    return output