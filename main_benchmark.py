import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from helper_funcs import *
from dataset import *
import gradio as gr
from load_autoencoder import autoencoder


print('loading data')
X_train, X_val = load_data('/workspaces/Autoencoder-Dedup/cifar_data')
print('loaded data')
dataset = Dataset()
path = "/workspaces/Autoencoder-Dedup/cifar__sample_data"
dataset.read_data_from_array(X_val[:1000])
dataset.inject_duplicates(20)
dataset.prepare_duplicates(autoencoder, 0.7)
dataset.benchmark(dataset.ground_truth)