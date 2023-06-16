import os
from os.path import exists
from tensorflow import keras
from keras.datasets import cifar10
import numpy as np

if not exists('cifar_data'):
    curr_dir = os.path.dirname(__file__) # directory of script
    p = r'{}/cifar_data'.format(curr_dir) # path to be created
    try:
        os.makedirs(p)
    except OSError:
        pass
    (X_train, _), (X_val, _) = keras.datasets.cifar10.load_data()
    X_train = X_train / 255.
    X_val = X_val / 255.
    np.save('cifar_data/X_train', X_train)
    np.save('cifar_data/X_val', X_val)