import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from helper_funcs import *
from dataset import *
import gradio as gr
from load_autoencoder import autoencoder


X_train, X_val = load_data('/workspaces/Autoencoder-Dedup/cifar_data')
dataset = Dataset()
dataset.read_data_from_array(X_val[:5000])
dataset.prepare_duplicates(autoencoder, 0.7)
dataset.start_viz()
