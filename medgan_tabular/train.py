from packages import *
from models import Generator, Discriminator, define_autoencoder, define_discriminator, define_generator
from utils import discriminator_loss, generator_loss

def train(data_seq, gen_model, dis_model, latent_dim, epochs = 2000, batch_size = 32
          , buffer_size = 16000, balance= 2, lr= 0.001):

	#converts data to tf dataset 
	dataset = tf.data.Dataset.from_tensor_slices(data_seq)
	train_dataset = dataset.shuffle(buffer_size).batch(batch_size)

	#initializes every brick
	generator = Generator()
	generator.define_model(gen_model)
	discriminator = Discriminator()
	discriminator.define_model(dis_model)
	gen_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
	disc_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

	#loss used for gan optimization
	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False) #from_logits -> doc

	@tf.function
	def train_step(input_datas, train_disc_only):
		noise = tf.random.normal([batch_size, latent_dim]) #new noise

		if train_disc_only: #we only update the discriminator's weights
			with tf.GradientTape(persistent=True) as tape:
				generated_tab = generator(noise)
				real_output = discriminator(input_datas)
				generated_output = discriminator(generated_tab)
				disc_loss = discriminator_loss(cross_entropy, real_output, generated_output) + sum(generator.losses) + sum(discriminator.losses)

			grad_disc = tape.gradient(disc_loss, discriminator.trainable_variables)
			disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))


		else: #we update both the discriminator and the generator-decoder
			with tf.GradientTape(persistent=True) as tape:
				generated_tab = generator(noise)
				real_output = discriminator(input_datas)
				generated_output = discriminator(generated_tab)

				gen_loss = generator_loss(cross_entropy, generated_output) + sum(generator.losses) + sum(discriminator.losses)
				disc_loss = discriminator_loss(cross_entropy, real_output, generated_output) + sum(generator.losses) + sum(discriminator.losses)

			grad_disc = tape.gradient(disc_loss, discriminator.trainable_variables)
			grad_gen = tape.gradient(gen_loss, generator.trainable_variables) #takes decoder into account as it's now intregrated in g model

			disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))
			gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))

		return gen_loss, disc_loss


	losses_gen, losses_disc = [],[]
	    
	for epoch in tqdm(range(balance*epochs)): 
	#balance stands for the number of gradient descent for the discriminator for one gradient descent of the generator

		start = time.time()
		total_gen_loss,total_disc_loss = 0,0

		for inputs in train_dataset: #iteration batch by batch
			gen_loss, disc_loss = train_step(inputs, train_disc_only= (epoch%(balance-1) != 0))

			total_gen_loss+= gen_loss
			total_disc_loss+= disc_loss
            
		if epoch%10==0:
			losses_gen.append(total_gen_loss)
			losses_disc.append(total_disc_loss)

	return losses_gen, losses_disc, generator, discriminator
    
#if __name__ == "__main__":
#    train()