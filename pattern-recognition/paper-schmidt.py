import matplotlib.pyplot as plt
import WesadDataManager as wdm

"""
- classifiers: benchmark: Decision Tree (DT), Random Forest
(RF), AdaBoost (AB), Linear Discriminant Analysis (LDA), and
k-Nearest Neighbour (kNN).
- binary and 3-classes classification
- The length of slide was 0.25 seconds and window size of 60 seconds.
"""
if __name__ == '__main__':
    db = wdm.WesadDataManager('WESAD')
    features = db.prepare_joined_data(wdm.WRIST, 60, 0.25)
        
    print ('=== FEATURES ===')
    print(features)
    wdm.plot_signals_from_df(db.processed_data, 'EDA_mean')
    
    