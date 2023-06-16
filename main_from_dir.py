import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from helper_funcs import *
from dataset import *
import gradio as gr
from load_autoencoder import autoencoder


dataset = Dataset()
path = "/workspaces/Autoencoder-Dedup/cifar__sample_data"
dataset.read_data_from_dir(path)
dataset.prepare_duplicates(autoencoder, 0.7)
print(dataset.data.shape, len(dataset.original_imgs))
new_data = dataset.remove_idxs_and_get_data([0, 1])
print(len(new_data))