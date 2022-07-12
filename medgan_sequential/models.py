from packages import *
from custom_tf_layers import *

#hercules : autoencoder

def define_hercules(NUM_CODE=1569, ori_dim=1571, embed_dim=64):
    x_input = layers.Input(shape= (None,None), ragged=True)

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

    flattened = RaggedDense(fromdim=ori_dim, todim=embed_dim)(divided)

    activated = layers.TimeDistributed(layers.ReLU())(flattened)

    densed = RaggedDense(fromdim=embed_dim, todim=ori_dim)(activated)

    soft_activated = layers.TimeDistributed(layers.Activation('sigmoid'))(densed)

    vectorized_output = layers.Lambda(lambda x: x.to_tensor())(soft_activated)

    hercules = models.Model(inputs=[x_input], outputs=[vectorized_output, vectorized_tab_ehrs])
    return hercules

#generator

def define_generator(pretrained_decoder_model, embed_dim=64, MAX_TIMESTEPS=257):
    randomDim = embed_dim
    @tf.function
    def slice_unitary(tuple):
        return tf.RaggedTensor.from_tensor(tuple[0][:tuple[1][0]])

    def slice_all(tuple):
        return tf.map_fn(slice_unitary, tuple,
            fn_output_signature=tf.RaggedTensorSpec(shape=(None,embed_dim), dtype=tf.float32))

    rd_input= layers.Input(shape= (MAX_TIMESTEPS,randomDim,), name= 'generator_rd_input_noise')
    length_conditionnal_input = layers.Input(shape= (1,), name= 'generator_length_conditional_input', dtype=tf.int32)
    
    dynamic = layers.Lambda(slice_all)([rd_input, length_conditionnal_input])

    fixed_last_dim_dynamic = layers.TimeDistributed(layers.Lambda(lambda x:x.to_tensor()))(dynamic)

    ##1st lstm module
    lstm = layers.LSTM(embed_dim,use_bias=False,return_sequences=True,
                    kernel_regularizer=l2(0.001),recurrent_regularizer=l2(0.001))(dynamic)
    batchnorm_0 = layers.TimeDistributed(
        layers.BatchNormalization(beta_regularizer=l2(0.001), gamma_regularizer=l2(0.001)), name='TDbn_lstm')(lstm)
    shortcut = layers.Add(name='TDshortcut')([batchnorm_0, fixed_last_dim_dynamic])

    #2nd module
    lstm = layers.LSTM(embed_dim,use_bias=False,return_sequences=True,
                    kernel_regularizer=l2(0.001),recurrent_regularizer=l2(0.001))(shortcut)
    batchnorm = layers.TimeDistributed(
        layers.BatchNormalization(beta_regularizer=l2(0.001), gamma_regularizer=l2(0.001)), name='TDbn_lstmbis')(lstm)
    shortcut_f = layers.Add(name='TDshortcutbis')([batchnorm, shortcut])

    #Dense preparation
    dense = RaggedDense(fromdim=embed_dim, todim=embed_dim)(shortcut_f)
    activation = layers.TimeDistributed(
        layers.Activation(tf.nn.relu, name='TDrelu'))(dense)

    #Decode generation from latent space to regular space
    decoded_output = pretrained_decoder_model(activation)

    model_generator = models.Model(inputs= [rd_input, length_conditionnal_input], outputs= [decoded_output])
    return model_generator

#discriminator

def define_preprocess(NUM_CODE):
    input_toprocess = layers.Input(shape= (None, None), ragged= True, name= 'discriminator_input_toformat')
    ehr = layers.Lambda(lambda rt: rt[:,:,1:])(input_toprocess)
    encoded_ehr = layers.TimeDistributed(layers.CategoryEncoding(num_tokens=NUM_CODE+2, output_mode='multi_hot'))(ehr)

    preprocess_model = models.Model(inputs= [input_toprocess], outputs= [encoded_ehr])
    return preprocess_model

def define_discriminator(NUM_CODE, MAX_TIMESTEPS, EMBEDIM):
    
    input_processed = layers.Input(shape=(None,NUM_CODE+2), ragged=True)
    
    lstm = layers.LSTM(EMBEDIM,
                    use_bias=False,
                    kernel_regularizer=l2(0.001),
                    recurrent_regularizer=l2(0.001),
                    return_sequences=True)(input_processed)

    tddense = layers.TimeDistributed(layers.Dense(8))(lstm)

    tdactivation = layers.TimeDistributed(layers.Activation(tf.nn.relu))(tddense)

    lstm_predictor = layers.LSTM(1,
                    use_bias=False,
                    activation= 'sigmoid',
                    kernel_regularizer=l2(0.001),
                    recurrent_regularizer=l2(0.001),
                    return_sequences=False)(tdactivation)

    model_discriminator = models.Model(inputs= [input_processed], outputs= [lstm_predictor])
    return model_discriminator