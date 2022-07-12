from packages import *
from utils import discriminator_loss, generator_loss

## models definition

def define_autoencoder(inputDim, embeddingDim, aeActivation, randomDim, dataType= ''):
	'''builds medgan-like autoencoder given the dimensions of input and random, the embedding dimension and the autoencoder activation'''

	x_input = tf.keras.layers.Input(shape=(inputDim,),
								name='input_encoder')

	x_dense = tf.keras.layers.Dense(embeddingDim, use_bias= True,
								kernel_regularizer= tf.keras.regularizers.l2(0.001),
								bias_regularizer= tf.keras.regularizers.l2(0.001),
								name=f'dense_encoder')(x_input)

	x_activation = tf.keras.layers.Activation(aeActivation,
								name= f'activation_encoder')(x_dense)

	encoder_model_default = tf.keras.models.Model(inputs=[x_input], outputs= x_activation)


	x_input = tf.keras.layers.Input(shape=(randomDim,),
								name='input_decoder')

	x_dense = tf.keras.layers.Dense(inputDim, use_bias= True, 
								kernel_regularizer= tf.keras.regularizers.l2(0.001),
								bias_regularizer= tf.keras.regularizers.l2(0.001),
								name=f'dense_decoder')(x_input)

	x_activation = tf.keras.layers.Activation('sigmoid' if dataType == 'binary' else 'relu',
								name= f'activation_decoder')(x_dense)

	decoder_model_default = tf.keras.models.Model(inputs=[x_input], outputs= x_activation)


	x_latent = encoder_model_default.output
	x_ae = decoder_model_default(x_latent)

	autoencoder_model_default = tf.keras.models.Model(inputs= [encoder_model_default.input], outputs= x_ae)

	return autoencoder_model_default, encoder_model_default, decoder_model_default


def define_generator(randomDim, generatorDims, is_trainable, trained_decoder, dataType= ''):
	'''builds generator medgan-like'''

	x_input = tf.keras.layers.Input(shape= (randomDim,),
								name='shortcut_-1')
	x_shortcut = x_input

	#print(generatorDims)#debug

	for idx, hidden_layer_dimension in enumerate(generatorDims[:-1]):
		x_dense = tf.keras.layers.Dense(hidden_layer_dimension, use_bias= False,
								kernel_regularizer= tf.keras.regularizers.l2(0.001),
								name=f'dense_{idx}')(x_shortcut)
		x_batchnorm = tf.keras.layers.BatchNormalization(trainable= is_trainable,
								beta_regularizer= tf.keras.regularizers.l2(0.001),
								gamma_regularizer= tf.keras.regularizers.l2(0.001),
								name=f'batchnorm_{idx}')(x_dense)
		x_activation = tf.keras.layers.Activation(tf.nn.relu,
								name=f'activation_{idx}')(x_batchnorm)
		x_shortcut = tf.keras.layers.Add(
								name=f'shortcut_{idx}-shortcut_{idx-1}-activation_{idx}-')([x_activation, x_shortcut])

	x_dense = tf.keras.layers.Dense(generatorDims[-1], use_bias= False,
								kernel_regularizer= tf.keras.regularizers.l2(0.001),
								name=f'dense_output')(x_shortcut)
	x_batchnorm = tf.keras.layers.BatchNormalization(trainable= is_trainable,
								name='batchnorm_output')(x_dense)
	x_activation = tf.keras.layers.Activation(tf.nn.tanh if dataType == 'binary' else tf.nn.relu,
								name='activation_output')(x_batchnorm)
	x_shortcut = tf.keras.layers.Add(
								name=f'shortcut_output-shortcut_{idx}-activation_output-')([x_activation, x_shortcut])

	x_output = trained_decoder.call(x_shortcut)

	generator_model_default = tf.keras.models.Model(inputs=[x_input],outputs=x_output)

	return generator_model_default


def define_discriminator(inputDim, discriminatorDims, discriminatorActivation, keepRate):
	'''builds discriminator medgan-like'''

	x_input = tf.keras.layers.Input(shape=(2*inputDim,),
								name='dropout_-1')
	x_dropout = x_input
	previous_dim = 2*inputDim

	for idx, hidden_layer_dimension in enumerate(discriminatorDims[:-1]):
						
		x_dense = tf.keras.layers.Dense(hidden_layer_dimension, use_bias= True,
								kernel_regularizer= tf.keras.regularizers.l2(0.001),
								bias_regularizer= tf.keras.regularizers.l2(0.001),
								name=f'dense_{idx}')(x_dropout)

		x_activation = tf.keras.layers.Activation(discriminatorActivation,
								name= f'activation_{idx}')(x_dense)

		x_dropout = tf.keras.layers.Dropout(rate=1-keepRate,
								name= f'dropout_{idx}')(x_activation)
		previous_dim = hidden_layer_dimension

	x_output = tf.keras.layers.Dense(1, use_bias= True,
								kernel_regularizer= tf.keras.regularizers.l2(0.001),
								bias_regularizer= tf.keras.regularizers.l2(0.001),
								name='dense_output')(x_dropout)

	y_hat = tf.keras.layers.Activation(tf.nn.sigmoid,
								name= f'activation_output')(x_output)

	discriminator_model_default = tf.keras.models.Model(inputs=[x_input],outputs=y_hat)

	return discriminator_model_default




#Classes

class Generator(tf.keras.Model):
	def __init__(self):
		super(Generator, self).__init__()

	def define_model(self, model):
		self.model = model
		
	def call(self, inputs):
		return self.model.call(inputs)


class Discriminator(tf.keras.Model):
	def __init__(self):
		super(Discriminator, self).__init__()

	def define_model(self, model):
		self.model = model
		
	def call(self, inputs):
		batchSize = inputs.shape[0]
		inputMean = tf.reshape(tf.tile(tf.reduce_mean(inputs, 0), [batchSize]), (batchSize, inputs.shape[-1]))
		inputswithmean = tf.concat([inputs, inputMean], 1)
		return self.model.call(inputswithmean)




#if __name__ == '__main__':#

#    parser = argparse.ArgumentParser()
#    args = parse_arguments(parser)