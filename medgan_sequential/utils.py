from packages import *

def customMultiBCE(y_true, y_pred, missed_penalization=tf.constant(1., dtype= tf.float32)):
    binary_true = tf.stack([missed_penalization*y_true, 1-y_true], axis=-1)
    binary_pred = tf.stack([y_pred, 1.- y_pred], axis=-1)
    respred = tf.reshape(binary_pred, (-1, 2))
    restrue = tf.reshape(binary_true, (-1,2))
    restrue = tf.cast(restrue, tf.float32)
    lgrespred = tf.math.log_sigmoid(respred)
    loss = tf.reduce_sum(-tf.multiply(restrue, lgrespred))
    return loss
#ex
#y_true = tf.constant([[1,0,1]])
#y_pred = tf.constant([[0.6,0.7,0.4]])

def scientific_writing(float, round_=None):
    if float is None :
        return None
    order = len(str(int(float)))
    x = float/10**(order-1) if round_ is None else round(float/10**(order-1), round_)
    return f'{x}e{order-1}'