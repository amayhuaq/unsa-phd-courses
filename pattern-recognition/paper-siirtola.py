#!/usr/bin/env python2
"""
Script to execute the implementation of Siirtola

Usage:
    paper-siirtola.py [-r EXECPART]

    paper-siirtola.py (-h | --help)
    paper-siirtola.py --version

Options:
    -r EXECPART         Part of paper to execute (1: best window size, 2: training models) [Default: '2']
    -h --help           Show this screen.
    --version           Show version.

Example:
    python paper-siirtola.py -r 1
"""

from docopt import docopt
import pandas as pd
import numpy as np
import WesadDataManager as wdm
import classifiers as clf
import multiprocessing as mp


windows = [15, 30, 60, 90, 120]
set_features = [
    ['EDA'],
    ['BVP'],
    ['TEMP'],
    ['ACC'],
    ['EDA', 'BVP'],
    ['EDA', 'TEMP'],
    ['TEMP', 'BVP'],
    ['EDA', 'TEMP', 'BVP'],
    ['ACC', 'EDA', 'TEMP', 'BVP']
]

RUNS = 5


# According to paper, select LDA to evaluate window sizes
def find_better_time(db):
    accuracies = {'mean':[], 'std':[]}
    
    for secs in windows:
        print ('===== window: ' + str(secs))
        features = db.prepare_joined_data(wdm.WRIST, secs, 0.25, binary=True)
        local_acc = []
        
        # LOSO
        for sid in wdm.list_subjects:
            test_data = features[features['subject'] == 'S' + str(sid)]
            train_data = features[features['subject'] != 'S' + str(sid)]
            train_data = train_data.sample(frac=1).reset_index(drop=True)
            # separating data and labels
            ytrain = train_data['label'].tolist()
            Xtrain = train_data.drop(columns=['subject', 'label']).values.tolist()    
            ytest = test_data['label'].tolist()
            Xtest = test_data.drop(columns=['subject', 'label']).values.tolist()
            # applying classifier
            print(np.shape(Xtrain), np.count_nonzero(Xtrain))
            print(np.shape(ytrain), np.count_nonzero(ytrain))
            print(np.shape(Xtest), np.count_nonzero(Xtest))
            print(np.shape(ytest), np.count_nonzero(ytest))
            accu = clf.lda_classifier(Xtrain, ytrain, Xtest, ytest)
            local_acc.append(accu)
        
        accuracies['mean'].append(np.mean(local_acc))
        accuracies['std'].append(np.std(local_acc))
                
    df = pd.DataFrame(accuracies)
    df.to_csv('../siirtola_bestWindow.csv')
    
    return windows[np.argmax(accuracies['mean'])]


def exec_set_features(all_features, sf):
    selCols = wdm.select_columns(all_features.columns, sf)
    selCols.append('subject')
    selCols.append('label')
    features = all_features[selCols]
    accuracies = {}
    df = None
    i = 0
    while i < RUNS:
        accuracies['lda'] = []
        accuracies['qda'] = []
        accuracies['rf'] = []
        
        # LOSO
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
            aclda = clf.lda_classifier(Xtrain, ytrain, Xtest, ytest)
            acqda = clf.qda_classifier(Xtrain, ytrain, Xtest, ytest)
            acrf = clf.rf_classifier(Xtrain, ytrain, Xtest, ytest)
            
            accuracies['lda'].append(aclda)
            accuracies['qda'].append(acqda)
            accuracies['rf'].append(acrf)
        
        df = pd.DataFrame(accuracies)
        i += 1
        df.to_csv('../siirtola_' + '-'.join(sf) + '_' + str(i) + '.csv')
    return 1
    

if __name__ == '__main__':
    args = docopt(__doc__, version='1.0')
    mod_exec = args['-r']
    
    db = wdm.WesadDataManager('WESAD')
    
    if mod_exec == '1':   # find best window size
        bWin = find_better_time(db)
        print('bestWin: ' + str(bWin))
    elif mod_exec == '2': # training models
        # Start evaluation with the best window size (120)
        all_features = db.prepare_joined_data(wdm.WRIST, 120, 0.25, binary=True)
        all_features.to_csv('../features_120.csv')
        #pool = mp.Pool(mp.cpu_count() - 1)
        for sf in set_features:
            #pool.apply(exec_set_features, args=(all_features, sf))
            exec_set_features(all_features, sf)
        #pool.close()
