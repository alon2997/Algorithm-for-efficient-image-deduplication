from keras import layers
import numpy as np

def random_zoom(im):
    return layers.RandomZoom(.2, .2)(im).numpy()
def random_flip(im):
    return layers.RandomFlip("horizontal")(im).numpy()
def random_rotation(im):
    return layers.RandomRotation(0.02)(im).numpy()

def aug_pipeline(im):
    first, second, third, fourth = np.random.randint(0, 2, 4)
    if first: im = random_zoom(im)
    elif second: im = random_flip(im)
    elif third: im = random_rotation(im)
    return im