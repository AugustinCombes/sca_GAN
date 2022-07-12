from packages import *
EMBEDIM = 64

class CutSigmoidOutput(layers.Layer):
    def __init__(self, threshold=1e-5, **kwargs):
        super(CutSigmoidOutput, self).__init__()
        self.threshold = tf.constant(threshold, tf.float32)
        
    def call(self, inputs):
        minus = inputs-self.threshold
        relu_activation = tf.nn.relu(minus)
        return relu_activation
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
class DynamicRepeatVector(layers.Layer):
    def __init__(self, **kwargs):
        super(DynamicRepeatVector, self).__init__()
        self.fun = lambda tuple : tf.RaggedTensor.from_tensor(tf.repeat(tf.expand_dims(tuple[0],0), tuple[1], axis=0))
        
    def call(self, inputs):
        toslice, slicer = inputs
        return tf.map_fn(self.fun, [toslice, slicer], 
          fn_output_signature=tf.RaggedTensorSpec(shape=(None, EMBEDIM), dtype=tf.float32))
    
class DynamicSlicer(layers.Layer):
    def __init__(self, **kwargs):
        super(DynamicSlicer, self).__init__()
        #self.fun = lambda tuple: tf.RaggedTensor.from_tensor(tuple[0][:tuple[1]])
        @tf.function
        def fun(tuple): 
            return tf.RaggedTensor.from_tensor(tuple[0][:tuple[1]])
        self.fun = fun
        
    def call(self, inputs):
        toslice, slicer = inputs
        slicer = tf.squeeze(slicer, -1)
        return tf.map_fn(self.fun, [toslice, slicer], 
          fn_output_signature=tf.RaggedTensorSpec(shape=(None, EMBEDIM), dtype=tf.float32))

class Slicer(layers.Layer):
    def __init__(self, **kwargs):
        super(Slicer, self).__init__()
        #self.fun = lambda tuple : tf.RaggedTensor.from_tensor(tf.repeat(tf.expand_dims(tuple[0],0), tuple[1], axis=0))
        @tf.function
        def fun(tuple) : 
            return tf.RaggedTensor.from_tensor(tf.repeat(tf.expand_dims(tuple[0],0), tuple[1], axis=0))
        self.fun = fun
        
    def call(self, inputs):
        toslice, slicer = inputs
        return tf.map_fn(self.fun, [toslice, slicer], 
          fn_output_signature=tf.RaggedTensorSpec(shape=(None, EMBEDIM), dtype=tf.float32))

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

#former bias-less version used for original training
#class RaggedDense(layers.Layer):
#    def __init__(self, fromdim, todim, bias=True, **kwargs):
#        super(RaggedDense, self).__init__()
#        self.fromdim = fromdim
#        self.todim = todim
#        self.kernel = tf.Variable(tf.random_normal_initializer()(shape=(self.fromdim,self.todim), dtype= tf.float32))    
#    def call(self, inputs):
#        res =  tf.tensordot(
#            inputs.to_tensor(), 
#            self.kernel, 
#            axes= [[-1], [0]])
#        return tf.RaggedTensor.from_tensor(res, padding= tf.zeros(shape=(self.todim,), dtype=tf.float32))

class RaggedDense(layers.Layer):
    def __init__(self, fromdim, todim, bias=False, **kwargs):
        super(RaggedDense, self).__init__()
        self.fromdim = fromdim
        self.todim = todim
        self.bias_bool = bias
        self.kernel = tf.Variable(tf.random_normal_initializer()(shape=(self.fromdim,self.todim), dtype= tf.float32))
        if self.bias_bool :
            self.bias = tf.Variable(tf.random_normal_initializer()(shape=(self.todim,), dtype= tf.float32))
    
    def call(self, inputs):
        res =  (tf.add(self.bias,
            tf.tensordot(
            inputs.to_tensor(), 
            self.kernel, 
            axes= [[-1], [0]])) if self.bias_bool
            else  tf.tensordot(
                inputs.to_tensor(), 
                self.kernel, 
                axes= [[-1], [0]]))
        return tf.RaggedTensor.from_tensor(res, padding= tf.zeros(shape=(self.todim,), dtype=tf.float32))