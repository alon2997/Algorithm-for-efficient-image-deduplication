import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import numpy as np
from keras.models import load_model
import os
from os.path import exists
from annoy_index import *
from helper_funcs import *
from dataset_obj import *
from autoencoder_helpers import *
from PIL import Image

X_train, X_val = load_data('/workspaces/Autoencoder-Dedup/cifar_data')

if not exists('cifar_sample_data'):
    curr_dir = os.path.dirname(__file__) # directory of script
    p = r'{}/cifar__sample_data'.format(curr_dir) # path to be created
    try:
        os.makedirs(p)
    except OSError:
        pass
    for i in range(100):
        im = Image.fromarray((X_train[i] * 255).astype(np.uint8))
        im.save(f"/workspaces/Autoencoder-Dedup/cifar__sample_data/cifar_{i}.jpeg", 'JPEG')
    for i in range(10):
        im = Image.fromarray((X_train[i] * 255).astype(np.uint8))
        im.save(f"/workspaces/Autoencoder-Dedup/cifar__sample_data/cifar_{i}_dup.jpeg", 'JPEG')        
    for i in range(1):
        im = Image.fromarray((X_train[i] * 255).astype(np.uint8))
        im = im.resize((64,64))
        im.save(f"/workspaces/Autoencoder-Dedup/cifar__sample_data/cifar_{i}_64x64.jpeg", 'JPEG')