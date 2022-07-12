from custom_tf_layers import *
from models import define_hercules
from utils import scientific_writing, customMultiBCE

def pretrain_autoencoder(hercules, ehr_dataset, BUFFER_SIZE, BATCH_SIZE, N_EPOCHS):
    perf_loss = customMultiBCE(hercules.layers[-1].output, hercules.layers[-2].output, missed_penalization=tf.constant(392.5, dtype=tf.float32))
    hercules.add_loss(perf_loss)
    ds_med = tf.data.Dataset.from_tensor_slices(ehr_dataset).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    result_losses = []

    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(input_batch):
        with tf.GradientTape() as tape:
            londons = hercules.call(input_batch)
            loss = sum(hercules.losses)
            
        gradients = tape.gradient(loss, hercules.trainable_variables)
        optimizer.apply_gradients(zip(gradients, hercules.trainable_variables))
        
        return loss, hercules.losses

    pbar = tqdm(range(N_EPOCHS))
    last_loss = None
    last_displayed_loss,displayed_batch_loss = None, None
    num_batch = int(ehr_dataset.shape[0]/BATCH_SIZE)+1

    for epoch in pbar :
        
        mean_loss = []
        for id_batch,batch in enumerate(ds_med) :
            batch_loss,_ = train_step(batch)
            
            pbar.set_description(f'Epoch loss: {scientific_writing(None if last_displayed_loss is None else last_displayed_loss.numpy(),4)}\
                                /Batch loss: {scientific_writing(None if displayed_batch_loss is None else displayed_batch_loss,4)} ({id_batch}/{num_batch})')
            mean_loss.append(batch_loss)
            displayed_batch_loss = batch_loss.numpy()
            
        last_loss = sum(mean_loss)/len(mean_loss)
        last_displayed_loss = last_loss
        result_losses.append(last_loss.numpy())
        
        test_result_mean = hercules(batch)[0].numpy().sum()
        input_reference_mean = tf.reduce_sum(tf.cast(tf.cast(batch[:,:,1:][0], tf.bool), tf.float32))
        
        pbar.write((
            f'\nEpoch {epoch} : '
            f'\nMeans synth res, true data : {(scientific_writing(test_result_mean,3),scientific_writing(input_reference_mean.numpy()))}'
            f'\nQuotient means res / min input : {test_result_mean/input_reference_mean}'
                ))