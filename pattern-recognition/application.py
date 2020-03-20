import matplotlib.pyplot as plt
import WesadDataManager as wdm
  
    
if __name__ == '__main__':
    db = wdm.WesadDataManager('WESAD', 'S4')
    features = {'c':{}, 'w':{}}
    features['w'] = db.prepare_joined_data(wdm.WRIST, binary=True, window_size=60)
    #features['c'] = db.prepare_joined_data(wdm.CHEST, binary=True, lfeatures=['Resp'])
        
    #print ('=== FEATURES ===')
    #print(features['w'])
    
    data = features['w'][features['w']['subject'] == 'S4']
    #print(data)
    plt.plot(data['label'].tolist())
    wdm.plot_signals_from_df(db.processed_data, 'BVP_hr_mean')
    