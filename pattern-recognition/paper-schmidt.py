import matplotlib.pyplot as plt
import pandas as pd
import WesadDataManager as wdm
import classifiers as clf

RUNS = 1
set_features = [
    { wdm.CHEST: ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp'], wdm.WRIST: ['ACC', 'EDA', 'TEMP', 'BVP']}
]


if __name__ == '__main__':
    db = wdm.WesadDataManager('WESAD')
    for sf in set_features:
        accuracies = {}
        df = None
        i = 0
        while i < RUNS:
            features1 = None
            features2 = None
            
            if len(sf[wdm.WRIST]) > 0:
                features1 = db.prepare_joined_data(wdm.WRIST, 60, 0.25, binary=True, lfeatures=sf[wdm.WRIST])
            if len(sf[wdm.CHEST]) > 0:
                features2 = db.prepare_joined_data(wdm.CHEST, 60, 0.25, binary=True, lfeatures=sf[wdm.CHEST])
            
            if features1 is None:
                features = features2
            elif features2 is None:
                features = features1
            else:
                features = features1.join(features2)
            print(features)
            print(features.columns)
            
            accuracies['dt'] = []
            accuracies['rf'] = []
            accuracies['ab'] = []
            accuracies['lda'] = []
            accuracies['knn'] = []
            
            # remove one subject
            for sid in wdm.list_subjects:
                test_data = features[features['subject'] == 'S' + str(sid)]
                train_data = features[features['subject'] != 'S' + str(sid)]
                train_data = train_data.sample(frac=1).reset_index(drop=True)
                # separating data and labels
                ytrain = train_data['label'].tolist()
                Xtrain = train_data.drop(columns=['subject', 'label']).values.tolist()    
                ytest = test_data['label'].tolist()
                Xtest = test_data.drop(columns=['subject', 'label']).values.tolist()
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
            
            tmp = pd.DataFrame(accuracies)
            t = [i for k in range(len(wdm.list_subjects))]
            tmp = tmp.join(pd.DataFrame({'time': t}))
            if df is None:
                df = tmp
            else:
                df = df.append(tmp, ignore_index = True)
            i += 1
        df.to_csv('../wesad_' + '-'.join(sf) + '.csv')