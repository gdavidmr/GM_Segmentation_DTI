import numpy as np
import pandas as pd

def data_prep_convert(dic):
    
    # concatenate GM and WM observations (voxels)
    array = np.concatenate((dic['lumbar_wm'],dic['lumbar_gm']),axis=0)
    
    # add an extra column with the labels
    labels = np.concatenate((np.zeros((dic['lumbar_wm'].shape[0],1)),np.ones((dic['lumbar_gm'].shape[0],1))),axis=0)
    array = np.append(array,labels,axis=1)
    
    # create a dataframe
    df = pd.DataFrame(array,columns=['fa','md','ad','rd','tissue'])
    
    return df
    