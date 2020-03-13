import matplotlib.pyplot as plt
import pandas as pd
import WesadDataManager as wdm
import classifiers as clf

"""
- classifiers: LDA (linear discriminant analysis), QDA (quadratic discriminant 
analysis), and RF (Random Forest).
- In this study, amusement and relaxed states were combined as one. 
Therefore, the studied problem was binary (stressed vs. non-stressed).
- For the model training signals were divided into windows, and from these windows,
features were extracted. To study the effect of window size,
different window sizes (15s, 30s, 60s, 90s, and 120s) were
compared. However, in each case, the length of slide was
the same, 0.25 seconds.
"""


if __name__ == '__main__':
    db = wdm.WesadDataManager('WESAD')
    windows = [15, 30, 60, 90, 120]
    accuracies = {}
    for secs in windows:
        features = db.prepare_joined_data(wdm.WRIST, secs, 0.25, binary=True)
            
        #print ('=== FEATURES ===')
        #print(features)
        #wdm.plot_signals_from_df(db.processed_data, 'EDA_mean')
        accuracies[str(secs) + '_lda'] = []
        accuracies[str(secs) + '_qda'] = []
        accuracies[str(secs) + '_rf'] = []
        
        # remove one subject
        for sid in wdm.list_subjects:
            test_data = features[features['subject'] == 'S' + str(sid)]
            train_data = features[features['subject'] != 'S' + str(sid)]
            # separating data and labels
            ytrain = train_data['label'].tolist()
            Xtrain = train_data.drop(columns=['subject', 'label']).values.tolist()    
            ytest = test_data['label'].tolist()
            Xtest = test_data.drop(columns=['subject', 'label']).values.tolist()
            # applying classifiers
            aclda = clf.lda_classifier(Xtrain, ytrain, Xtest, ytest)
            acqda = clf.qda_classifier(Xtrain, ytrain, Xtest, ytest)
            acrf = clf.rf_classifier(Xtrain, ytrain, Xtest, ytest)
            
            accuracies[str(secs) + '_lda'].append(aclda)
            accuracies[str(secs) + '_qda'].append(acqda)
            accuracies[str(secs) + '_rf'].append(acrf)
    
    df = pd.DataFrame(accuracies)
    df.to_csv('siirtola.csv')