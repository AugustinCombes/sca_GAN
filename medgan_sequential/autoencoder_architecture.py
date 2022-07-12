import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K

import tensorflow_probability as tfp

from tqdm.notebook import tqdm
from tqdm.keras import TqdmCallback

NUM_CODE,MAX_LENGTH = 1569,2359

N_PATIENTS = 12581

def plot_model(m, **kwargs):
    return tf.keras.utils.plot_model(m, show_shapes= True, show_layer_names= True, **kwargs)

#************************************************************************************************************************************
#load dataset

with open("weekwise_encoded_variantlength.pickle", "rb") as file:
    med = pickle.load(file)
    
for patient in med.keys():
    for week in med[patient].keys():
        tmp = list(set(med[patient][week]))
        med[patient][week] = [week] + tmp
    med[patient] = list(med[patient].values())
    
med = list(med.values())

med = tf.ragged.constant(med)

#************************************************************************************************************************************
#custom tf layers

class SelectSlice(layers.Layer):
    def __init__(self, slice_idx, **kwargs):
        super(SelectSlice, self).__init__()
        self.slice_idx = slice_idx
        
    def call(self, input):
        return tf.reduce_sum(input[:,:,self.slice_idx:(self.slice_idx+1)], axis=-1, keepdims=True)
    
class SelectSliceRange(layers.Layer):
    def __init__(self, slice_idx_min, slice_idx_max,**kwargs):
        super(SelectSliceRange, self).__init__()
        self.slice_idx_min = slice_idx_min
        self.slice_idx_max = slice_idx_max
    
    def call(self, input):
        if self.slice_idx_max is None :
            return input[:,:,self.slice_idx_min:]
        else :
            return input[:,:,self.slice_idx_min:self.slice_idx_max]

class RaggedDense(layers.Layer):
    def __init__(self, fromdim, todim, bias=True, **kwargs):
        super(RaggedDense, self).__init__()
        self.fromdim = fromdim
        self.todim = todim
        self.bias_bool = bias
        self.kernel = tf.Variable(tf.random_normal_initializer()(shape=(self.fromdim,self.todim), dtype= tf.float32))
        if self.bias_bool :
            self.bias = tf.Variable(tf.random_normal_initializer()(shape=(self.todim,), dtype= tf.float32))
    
    def call(self, inputs):
        res =  tf.add(self.bias,
            tf.tensordot(
            inputs.to_tensor(), 
            self.kernel, 
            axes= [[-1], [0]]))
        return tf.RaggedTensor.from_tensor(res, padding= tf.zeros(shape=(self.todim,), dtype=tf.float32))

#************************************************************************************************************************************
#define model
x_input = layers.Input(shape= (None,None), ragged=True)
#x_input = med[0:5]

row_split = layers.Lambda(lambda x: x.row_splits)
times = SelectSlice(0)(x_input)
ehrs = SelectSliceRange(1,None)(x_input)
reference_split = layers.Lambda(lambda x: x.nested_row_splits)(x_input)
reference_nested_lengths = layers.Lambda(lambda x: x.nested_row_lengths())(x_input)

tab_ehrs = layers.Lambda(lambda batch :
                            tf.cast(
                                tf.reduce_sum(
                                    tf.one_hot(indices=
                                        tf.cast(batch, tf.int32), 
                                     depth= NUM_CODE+2),
                                  axis=2),
                             tf.float32))(ehrs)

vectorized_tab_ehrs = layers.Lambda(lambda x: x.to_tensor())(tab_ehrs) #the goal is to retrieve this layer

divided = tab_ehrs

flattened = RaggedDense(fromdim=1571, todim=16)(divided)

activated = layers.TimeDistributed(layers.ReLU())(flattened)

densed = RaggedDense(fromdim=16, todim=1571)(activated)

soft_activated = layers.TimeDistributed(layers.Activation('sigmoid'))(densed)

vectorized_output = layers.Lambda(lambda x: x.to_tensor())(soft_activated)

hercules = models.Model(inputs=[x_input], outputs=[vectorized_output, vectorized_tab_ehrs])