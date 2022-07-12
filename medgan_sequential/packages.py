import pickle
import numpy as np
#import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

import tensorflow_probability as tfp

from tqdm.notebook import tqdm
from tqdm.keras import TqdmCallback

#overwrite tqdm plot model with other default arguments
def plot_model(m, **kwargs):
    return tf.keras.utils.plot_model(m, show_shapes= True, show_layer_names= True, **kwargs)