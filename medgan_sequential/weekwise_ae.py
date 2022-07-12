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

N_PATIENTS = 12581
NUM_CODE = 1569

def plot_model(m, **kwargs):
    return tf.keras.utils.plot_model(m, show_shapes= True, show_layer_names= True, **kwargs)


#Load database with count occurences (in IN) of ehr events, reduce it to indicatrices with list(set()) operation and format it to ragged tensor after adding the date in 1st position.

#Output ragged tensor rank is 3 : [N_PATIENTS (cst), week_index (ragged), week_date&encoded_events (ragged)]

with open("weekwise_encoded_variantlength.pickle", "rb") as file:
    med = pickle.load(file)
    
for patient in med.keys():
    for week in med[patient].keys():
        tmp = list(set(med[patient][week]))
        med[patient][week] = [week] + tmp #+ [len(tmp)]
    med[patient] = list(med[patient].values())
    
med = list(med.values())

#Put 0 as the int to predict, then in GAN loss we'll manage that if 0 is the 
#ehr code to be predicted, this code doesn't contribute to the loss
for patient in range(len(med)):
    if len(med[patient]) == 1 :
        med[patient].append([0,0])
        #print(med[patient])

med = tf.ragged.constant(med)

#Utils layers that may be to be replaced by layers.Lambda

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
class SelectSlice(layers.Layer):
    def __init__(self, slice_idx, **kwargs):
        super(SelectSlice, self).__init__()
        self.slice_idx = slice_idx
        
    def call(self, input):
        return tf.reduce_sum(input[:,:,self.slice_idx:(self.slice_idx+1)], axis=-1, keepdims=True)

#Homemade layer that allows to perform dense training on ragged layer without having to worry about converting to tensor, managing fill values and removing them. Note that the bias isn't yet developed.

class RaggedDense(layers.Layer):
    def __init__(self, fromdim, todim, bias=True, **kwargs):
        super(RaggedDense, self).__init__()
        self.fromdim = fromdim
        self.todim = todim
        #self.bias_bool = bias
        self.kernel = tf.Variable(tf.random_normal_initializer()(shape=(self.fromdim,self.todim), dtype= tf.float32))
        #self.bias = tf.Variable(tf.random_normal_initializer()(shape=(self.todim,), dtype= tf.float32))
    
    def call(self, inputs):
        res =  tf.tensordot(
            inputs.to_tensor(), 
            self.kernel, 
            axes= [[-1], [0]])
        return tf.RaggedTensor.from_tensor(res, padding= tf.zeros(shape=(self.todim,), dtype=tf.float32))

#Very tricky argmax-selecting layer
BATCH_SIZE = 32

@tf.function
def ones_zeros(nb_ones, total_dim):
    return tf.concat([tf.ones((nb_ones),tf.int32),tf.zeros((total_dim-nb_ones),tf.int32)], axis=0)

@tf.function
def clear(partially_ragged_row, lens_row):
    maped = tf.map_fn(lambda x: ones_zeros(x, 1571), lens_row)
    res = tf.multiply(partially_ragged_row, maped)
    return res

class ExtractLengths(layers.Layer):
    def __init__(self, **kwargs):
        super(ExtractLengths, self).__init__()
        
    def call(self, inputs):
        base = inputs[:,:,1:].nested_row_lengths()[1]
        base = tf.reshape(base, (1, tf.shape(base)[0]))
        ids = inputs.value_rowids()
        ids = tf.cast(ids, tf.int32)
        return [base,ids]

@tf.function
def map_clear_to_patients2(pr, le):
    malin = tf.map_fn(lambda t: clear(t[0], t[1]), [pr, tf.expand_dims(le, -1)], 
          fn_output_signature= tf.TensorSpec(dtype=tf.int32, shape=(1, 1571)))
    sq = tf.squeeze(malin)
    nozeros = tf.RaggedTensor.from_tensor(sq, padding=tf.constant(0,tf.int32))
    nozeros = tf.expand_dims(nozeros, 0)
    return nozeros

class CutOutput(layers.Layer):
    def __init__(self, **kwargs):
        super(CutOutput, self).__init__()
        
    def call(self, inputs, num_patient=BATCH_SIZE):
        to_cut, lengths_infos = inputs
        base, ids = lengths_infos
        
        argsorted_output = tf.argsort(-to_cut, axis=-1)
        
        partially_ragged = tf.RaggedTensor.from_tensor(argsorted_output, padding=tf.constant(np.arange(NUM_CODE+2)))
        
        ohe = tf.one_hot(indices=ids,
           depth=1+tf.reduce_max(ids))
        ohe = tf.cast(ohe, tf.int64)
        
        #num_patient = partially_ragged.shape[0] #impossible to input a dynamic shape in tf.repeat
        
        stackedbase = tf.repeat(base, num_patient, axis=0)
        
        mult= tf.multiply(tf.transpose(ohe), stackedbase)
        
        rt = tf.RaggedTensor.from_tensor(mult, padding=tf.constant(0, tf.int64))
        
        lens = tf.RaggedTensor.from_tensor(tf.reverse(rt, axis=[1]).to_tensor(), padding=tf.constant(0,tf.int64))
        lens = tf.cast(lens, tf.int32)
        lens = tf.reverse(lens, axis=[1])
        
        zl = tf.squeeze(tf.map_fn(lambda t: map_clear_to_patients2(
            t[0], 
            t[1]
                            ), 
                  [partially_ragged, lens],
                  fn_output_signature=tf.RaggedTensorSpec(dtype=tf.int32, shape=(1, None, None))),
                       axis=1)
            
        return zl

x_input = layers.Input(shape= (None,None), ragged=True)
#x_input = med[0:5]

row_split = layers.Lambda(lambda x: x.row_splits)
times = SelectSlice(0)(x_input)
ehrs = SelectSliceRange(1,None)(x_input)
#ehrs = SelectSliceRange(1,-1)(x_input)
#lens = SelectSlice(-1)(x_input)
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

vectorized_tab_ehrs = layers.Lambda(lambda x: x.to_tensor())(tab_ehrs) #to retrieve

divided = tab_ehrs

RD1 = RaggedDense(fromdim=1571, todim=64)
flattened = RD1(divided)

activated = layers.TimeDistributed(layers.ReLU())(flattened)

RD2 = RaggedDense(fromdim=64, todim=1571)
densed = RD2(activated)

soft_activated = layers.TimeDistributed(layers.Activation('sigmoid'))(densed)

vectorized_output = layers.Lambda(lambda x: x.to_tensor())(soft_activated)

input_reference_1 = ExtractLengths()(x_input)
ragged_int_encoded_output = CutOutput()([vectorized_output, input_reference_1])


hercules = models.Model(inputs=[x_input], outputs=[ragged_int_encoded_output])#retimes
#plot_model(hercules)

#Set weights to pretrained model on 235 epochs

w_1571_64 = np.load('hercules_235epochs/w_1571_64.npy')
w_64_1571 = np.load('hercules_235epochs/w_64_1571.npy')
print(w_1571_64.shape, w_64_1571.shape)
hercules.set_weights([tf.constant(w_1571_64), tf.constant(w_64_1571)])

#hercules.call(med[0:BATCH_SIZE])