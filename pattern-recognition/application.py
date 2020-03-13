import matplotlib.pyplot as plt
import WesadDataManager as wdm
  
    
if __name__ == '__main__':
    db = wdm.WesadDataManager('WESAD', 'S3')
    features = {'c':{}, 'w':{}}
    features['w'] = db.prepare_joined_data(wdm.WRIST, binary=True)
    #features['c'] = db.extract_basic_features(wdm.CHEST)
        
    print ('=== FEATURES ===')
    print(features['w'])
    
    data = features['w'][features['w']['subject'] == 'S3']
    print(data)
    plt.plot(data['label'].tolist())
    wdm.plot_signals_from_df(db.processed_data, 'EDA_mean')
    