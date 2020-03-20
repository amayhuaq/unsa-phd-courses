import matplotlib.pyplot as plt
import pandas as pd
import WesadDataManager as wdm
import classifiers as clf

RUNS = 5

set_features = [
    { wdm.CHEST: [], wdm.WRIST: ['ACC']},
    { wdm.CHEST: ['ACC'], wdm.WRIST: []},
    { wdm.CHEST: [], wdm.WRIST: ['BVP']},
    { wdm.CHEST: [], wdm.WRIST: ['EDA']},
    { wdm.CHEST: [], wdm.WRIST: ['TEMP']},
    { wdm.CHEST: [], wdm.WRIST: ['EDA', 'TEMP', 'BVP']},
    { wdm.CHEST: ['ECG'], wdm.WRIST: []},
    { wdm.CHEST: ['EDA'], wdm.WRIST: []},
    { wdm.CHEST: ['EMG'], wdm.WRIST: []},
    { wdm.CHEST: ['Resp'], wdm.WRIST: []},
    { wdm.CHEST: ['Temp'], wdm.WRIST: []},
    { wdm.CHEST: ['ECG', 'EDA', 'EMG', 'Resp', 'Temp'], wdm.WRIST: []},
    { wdm.CHEST: [], wdm.WRIST: ['ACC', 'EDA', 'TEMP', 'BVP']},
    { wdm.CHEST: ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp'], wdm.WRIST: []},
    { wdm.CHEST: ['ECG', 'EDA', 'EMG', 'Resp', 'Temp'], wdm.WRIST: ['EDA', 'TEMP', 'BVP']},
    { wdm.CHEST: ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp'], wdm.WRIST: ['ACC', 'EDA', 'TEMP', 'BVP']}
]


if __name__ == '__main__':
    db = wdm.WesadDataManager('WESAD')
    features1 = db.prepare_joined_data(wdm.WRIST, 60, 0.25, binary=True)
    features2 = db.prepare_joined_data(wdm.CHEST, 60, 0.25, binary=True)
    features1 = features1.drop(columns=['subject', 'label'])
    ids = {val : val + '_' + wdm.WRIST for val in features1.columns}
    features1 = features1.rename(columns=ids)
    ids = {val : val + '_' + wdm.CHEST for val in features2.columns}
    features2 = features2.rename(columns=ids)
    all_features = features1.join(features2)
    
    colSubName = 'subject_' + wdm.CHEST
    colLabName = 'label_' + wdm.CHEST
    
    s = 0
    for sf in set_features:
        if len(sf[wdm.WRIST]) == 0:
            selCols = wdm.select_columns(features2.columns, sf[wdm.CHEST])
        elif len(sf[wdm.CHEST]) == 0:
            selCols = wdm.select_columns(features1.columns, sf[wdm.WRIST])
        else:
            selCols1 = wdm.select_columns(features1.columns, sf[wdm.WRIST])
            selCols2 = wdm.select_columns(features2.columns, sf[wdm.CHEST])
            selCols = selCols1 + selCols2
        selCols.append(colSubName)
        selCols.append(colLabName)
        features = all_features[selCols]
        print(features)
        print(features.columns)
        
        accuracies = {}
        df = None
        i = 0
        
        while i < RUNS:
            accuracies['dt'] = []
            accuracies['rf'] = []
            accuracies['ab'] = []
            accuracies['lda'] = []
            accuracies['knn'] = []
            
            # remove one subject
            for sid in wdm.list_subjects:
                test_data = features[features[colSubName] == 'S' + str(sid)]
                train_data = features[features[colSubName] != 'S' + str(sid)]
                train_data = train_data.sample(frac=1).reset_index(drop=True)
                # separating data and labels
                ytrain = train_data[colLabName].tolist()
                Xtrain = train_data.drop(columns=[colSubName, colLabName]).values.tolist()    
                ytest = test_data[colLabName].tolist()
                Xtest = test_data.drop(columns=[colSubName, colLabName]).values.tolist()
                # applying classifiers
                acdt = clf.dt_classifier(Xtrain, ytrain, Xtest, ytest)
                acrf = clf.rf_classifier(Xtrain, ytrain, Xtest, ytest)
                acab = clf.ab_classifier(Xtrain, ytrain, Xtest, ytest)
                aclda = clf.lda_classifier(Xtrain, ytrain, Xtest, ytest)
                acknn = clf.knn_classifier(Xtrain, ytrain, Xtest, ytest)
                
                accuracies['dt'].append(acdt)
                accuracies['rf'].append(acrf)
                accuracies['ab'].append(acab)
                accuracies['lda'].append(aclda)
                accuracies['knn'].append(acknn)
            
            df = pd.DataFrame(accuracies)
            df.to_csv('../wesad_' + str(s) + '_' + str(i) + '.csv')
            i += 1
        s += 1
