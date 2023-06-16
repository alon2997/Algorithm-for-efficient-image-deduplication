import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from helper_funcs import *
from dataset import *
import gradio as gr
from load_autoencoder import autoencoder


X_train, X_val = load_data('/workspaces/Autoencoder-Dedup/cifar_data')
dataset = Dataset()
path = "/workspaces/Autoencoder-Dedup/cifar__sample_data"
dataset.read_data_from_array(X_val[:1000])
print(dataset.data.shape, dataset.original_imgs.shape)
dataset.prepare_duplicates(autoencoder, 1)
print(dataset.dup_components)
new_data = dataset.remove_all_dups_and_get_data()
print(new_data.shape)