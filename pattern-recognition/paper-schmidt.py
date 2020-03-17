import matplotlib.pyplot as plt
import pandas as pd
import WesadDataManager as wdm
import classifiers as clf

"""
- classifiers: benchmark: Decision Tree (DT), Random Forest
(RF), AdaBoost (AB), Linear Discriminant Analysis (LDA), and
k-Nearest Neighbour (kNN).
- binary and 3-classes classification
- The length of slide was 0.25 seconds and window size of 60 seconds.
"""
if __name__ == '__main__':
    db = wdm.WesadDataManager('WESAD')
    accuracies = {}
    
    features = db.prepare_joined_data(wdm.WRIST, 60, 0.25, binary=True)
        
    #print ('=== FEATURES ===')
    #print(features)
    #wdm.plot_signals_from_df(db.processed_data, 'EDA_mean')

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
    
    df = pd.DataFrame(accuracies)
    df.to_csv('../wesad.csv')