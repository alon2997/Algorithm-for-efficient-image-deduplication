import numpy as np 

def load_data(path):
    X_train = np.load(f'{path}/X_train.npy')
    X_val = np.load(f'{path}/X_val.npy')
    return (X_train, X_val)
