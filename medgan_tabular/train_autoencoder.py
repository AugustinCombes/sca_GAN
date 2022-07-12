from packages import *
from models import define_autoencoder

def train_autoencoder2(ae, input_tabular_data, epochs= 500, lr= 0.001):

	#rules of training for the autoencoder
	ae.compile(optimizer= tf.keras.optimizers.Adam(learning_rate= lr), loss=tf.losses.MeanSquaredError()) #used to be optimizer= 'adam'

	#actual training
	history = ae.fit(input_tabular_data, input_tabular_data, epochs= epochs, shuffle= True, 
	                              verbose= 0, callbacks= [TqdmCallback(verbose=1)], use_multiprocessing=True,
	                              workers= 3)

	return history

def train_autoencoder(ae, input_tabular_data, epochs= 500, lr= 0.001):
	batch_size = 100

	ae_optim = tf.keras.optimizers.Adam()#learning_rate=lr
	mse = tf.keras.losses.MSE

	@tf.function
	def pretrain_step(input_data):
		with tf.GradientTape(persistent=True) as tape:
			ae_loss = tf.reduce_sum(mse(input_data, ae.call(input_data)))
			ae_loss += sum(ae.losses)

		grad_ae = tape.gradient(ae_loss, ae.trainable_variables)
		ae_optim.apply_gradients(zip(grad_ae, ae.trainable_variables))

		return ae_loss

	nPretrainBatches = int(input_tabular_data.shape[0]/batch_size)
	pbar = tqdm(range(epochs))
	meanlossdisplay = 0

	for pretrainEpoch in pbar:
		pbar.set_description(f'Processing pretraining. Current mean batch loss : {round(meanlossdisplay,5)}')
		meanlossdisplay = 0

		idx = np.random.permutation(input_tabular_data.shape[0])

		for batch_id in range(nPretrainBatches):
			batch = tf.constant(input_tabular_data.numpy()[idx[batch_id*batch_size:(batch_id+1)*batch_size]])

			loss = pretrain_step(batch)
			meanlossdisplay += loss.numpy()

		meanlossdisplay/=nPretrainBatches