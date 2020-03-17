import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import feature_extractor as fex

devices_sampling = {
    'w': {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4},
    'c': {'ACC':700, 'ECG':700, 'EDA':700, 'EMG':700, 'Resp':700, 'Temp':700}
}

list_subjects = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]

list_functions = {
    'ACC': fex.get_acc_features,
    'ECG': fex.get_ecg_features,
    'BVP': fex.get_ecg_features,
    'EDA': fex.get_eda_features,
    'EMG': fex.get_emg_features,
    'TEMP': fex.get_temp_features,
    'Temp': fex.get_temp_features,
    'Resp': fex.get_resp_features
}

WRIST = 'w'
CHEST = 'c'

sensor_keys = {
    CHEST: ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp'],
    WRIST: ['ACC', 'BVP', 'EDA', 'TEMP']
}


def plot_signals_from_df(data, signal='label'):
    fig = plt.figure()
    for subj, vals in data.items():
        plt.plot(vals[signal].tolist(), label=subj)
    plt.title('Signal graph: ' + signal)
    plt.legend()
    plt.show()
 
    
class SubjectData:
    def __init__(self, path, subject):
        self.id = subject
        self.signal_data = {}
        with open(subject + '/' + subject + '.pkl', 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        self.labels = data['label']
        self.signal_data[WRIST] = data['signal']['wrist']
        self.signal_data[CHEST] = data['signal']['chest']
        self.wrist_data = data['signal']['wrist']
        self.chest_data = data['signal']['chest']

    def get_labels(self):
        """Sampled at 700 Hz. The following IDs are provided:
        0 = not defined / transient, 1 = baseline,
        2 = stress, 3 = amusement, 4 = meditation,
        5/6/7 = should be ignored in this dataset"""
        return self.labels

    def get_wrist_data(self):
        return self.signal_data[WRIST]

    def get_chest_data(self):
        return self.signal_data[CHEST]


class WesadDataManager:
    def __init__(self, db_path, subject=None):
        self.db_path = db_path
        self.obj_data = {}
        self.processed_data = {}

        os.chdir(db_path)
        if subject is None:
            self.load_data()
        else:
            self.load_data_one_subject(subject)

    def load_data_one_subject(self, subject):
        self.obj_data[subject] = SubjectData(self.db_path, subject)
        fex.print_obj_len(self.obj_data[subject].get_wrist_data(), subject)

    def load_data(self):
        for i in list_subjects:
            self.load_data_one_subject('S' + str(i))

    def extract_basic_features(self, obj, source=WRIST, window_size=1, shift=1, id_features=[]):
        cfeatures = pd.DataFrame({})
        for key2 in id_features:
            signal = obj.signal_data[source][key2]
            res = list_functions[key2](signal, devices_sampling[source][key2], window_size, shift)
            fex.print_obj_len(res, key2)
            # creating dataframe
            ids = {val : key2 + '_' + val for val, arr in res.items()}
            df = pd.DataFrame(res)
            df = df.rename(columns=ids)
            if (cfeatures.size == 0):
                cfeatures = df
            else:
                cfeatures = cfeatures.join(df)            
        return cfeatures
        
    def prepare_joined_data(self, source=WRIST, window_size=1, shift=1, binary=False, wfilter=True, lfeatures=None):
        df = pd.DataFrame({})
        if(lfeatures is None):
            lfeatures = sensor_keys[source]
        for key, obj in self.obj_data.items():
            features = self.extract_basic_features(obj, source, window_size, shift, lfeatures)
            labels = fex.get_final_labels(obj.labels, binary=binary)
            subject = [key for i in range(len(labels))]
            features = features.join(pd.DataFrame({'subject': subject}))
            features = features.join(pd.DataFrame({'label': labels}))
            if(wfilter):
                features = features[features['label'].isin([fex.BASELINE_STATE, fex.STRESS_STATE, fex.AMUSEMENT_STATE])]
            self.processed_data[key] = features
            if (df.size == 0):
                df = features
            else:
                df = df.append(features, ignore_index = True)
        return df
