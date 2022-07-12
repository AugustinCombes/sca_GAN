from packages import *
from models import *
from utils import *

def train(model_generator, model_discriminator, preprocess_model, med, EPOCHS, BUFFER_SIZE=16000, BATCH_SIZE=32, MAX_TIMESTEPS=257,
          BALANCE=1):
    
    med_dataset = tf.data.Dataset.from_tensor_slices(med).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def generator_perf_loss(unitary_loss, fake_output):
        return unitary_loss(tf.ones_like(fake_output), fake_output)

    def discriminator_perf_loss(unitary_loss, real_output, fake_output):
        real_loss = unitary_loss(tf.ones_like(real_output), real_output)
        fake_loss = unitary_loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss+fake_loss
        return total_loss

    gen_optimizer = tf.keras.optimizers.Adam()
    dis_optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(inputs, train_both):
        noise = tf.random.normal([BATCH_SIZE, MAX_TIMESTEPS, EMBEDIM])
        lens_poor_simulation = tf.constant(1,tf.int32) + tf.cast(
            tf.abs(tf.random.normal((BATCH_SIZE, 1), mean=66.106, stddev=38.185)), tf.int32)
        
        train_both = True #new, as the discriminator seems really strong compared to the generator
        
        if not train_both :
            with tf.GradientTape(persistent=True) as tape:
                generated = model_generator([noise, lens_poor_simulation])
                (generated_output, real_output) = (
                    model_discriminator(generated), 
                    model_discriminator(preprocess_model(inputs)))
                
                gen_loss = (generator_perf_loss(bce, generated_output)
                            +sum(model_generator.losses)+sum(model_discriminator.losses))
                dis_loss = (discriminator_perf_loss(bce, real_output, generated_output)
                            +sum(model_generator.losses)+sum(model_discriminator.losses))
                
            grad_disc = tape.gradient(dis_loss, model_discriminator.trainable_variables)
            
            dis_optimizer.apply_gradients(zip(grad_disc, model_discriminator.trainable_variables))
            
        else :
            with tf.GradientTape(persistent=True) as tape:
                generated = model_generator([noise, lens_poor_simulation])
                generated_output = model_discriminator(generated)
                real_output = model_discriminator(preprocess_model(inputs))
                
                gen_loss = (generator_perf_loss(bce, generated_output)
                            +sum(model_generator.losses)+sum(model_discriminator.losses))
                dis_loss = (discriminator_perf_loss(bce, real_output, generated_output)
                            +sum(model_generator.losses)+sum(model_discriminator.losses))
                
            grad_disc = tape.gradient(dis_loss, model_discriminator.trainable_variables)
            grad_gen = tape.gradient(gen_loss, model_generator.trainable_variables)
            
            dis_optimizer.apply_gradients(zip(grad_disc, model_discriminator.trainable_variables))
            gen_optimizer.apply_gradients(zip(grad_gen, model_generator.trainable_variables))
            
        return gen_loss, dis_loss, generated_output, real_output

    losses_gen, losses_dis = [], []
    epoch_pbar = tqdm(range(BALANCE*EPOCHS))
    total_gen_loss, total_dis_loss = 0,0

    lmax = []
    lmean = []
    hist_disc_pred = []

    for epoch in epoch_pbar:
        batch_pbar = tqdm(enumerate(med_dataset))
        epoch_pbar.set_description(f"Epoch {epoch}: GenLoss {total_gen_loss} - DisLoss {total_dis_loss}")
        total_gen_loss, total_dis_loss = 0,0
        disc_pred = []
        
        for idx,inputs in batch_pbar:
            if True:
                gen_loss, dis_loss, generated_output, real_output = train_step(inputs, train_both= (epoch%BALANCE)==0)
                total_gen_loss, total_dis_loss = total_gen_loss+gen_loss, total_dis_loss+dis_loss
                batch_pbar.set_description(f"Batch {idx}: GenLoss {gen_loss} - DisLoss {dis_loss}")
                disc_pred.append(list(generated_output.numpy()))
                disc_pred.append(list(real_output.numpy()))
        
        res = model_generator.call(
            [tf.random.normal([BATCH_SIZE, MAX_TIMESTEPS, EMBEDIM]),
            tf.cast(tf.abs(tf.random.normal((BATCH_SIZE, 1), mean=66.106, stddev=38.185)), tf.int32)])
                
        lmax.append(tf.reduce_max(res))
        lmean.append(tf.reduce_mean(res))

        losses_gen.append(total_gen_loss)
        losses_dis.append(total_dis_loss)
        hist_disc_pred.append(disc_pred)