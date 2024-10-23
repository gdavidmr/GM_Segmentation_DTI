# load mat file into a numpy array
from scipy.io import loadmat

def data_prep_load(data):
    dic = loadmat(data)
    return dic