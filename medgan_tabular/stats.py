from packages import *
from models import Generator, Discriminator, define_autoencoder, define_discriminator, define_generator
from utils import discriminator_loss, generator_loss

def simulate(generator_callable_model, sample_size= 10000):
    '''Generates sample_size synthetic data from generator_callable_model'''
    
    inp_dim = generator_callable_model.get_weights()[0][0].shape[0]
    base_noise = tf.random.normal([sample_size, inp_dim])
    gen = generator_callable_model.call(base_noise)
    
    return gen

def moment_dim_wise(generator_callable_model, source_data, sample_size= 10000, mo= 1):
    '''Generates sample_size synthetic data from generator_callable_model, and compares its moment 
    of order mo with the one of source_data'''
    
    gen = simulate(generator_callable_model, sample_size)
    sampled_real_data = tf.constant(pd.DataFrame(source_data.numpy()).sample(sample_size))
    
    synth, real = np.array(tf.round(gen), dtype = np.int64), sampled_real_data.numpy()
    synth, real = (synth.mean(axis=0)**mo)**(1/mo), (real.mean(axis=0)**mo)**(1/mo)
    
    return (synth, real)

def plot_dim_mo(synth, real, border= None):
	'''plots synthetic vs real data moment, with automatic well defined border to square up the plot'''

	if border is None:
		border = max(synth.max(), real.max())*1.01

	plt.ylim(0, border)
	plt.xlim(0, border)
	plt.plot(real, synth, 'ro')
	plt.plot(np.linspace(0,border,10000), np.linspace(0,border,10000), 'b')
	plt.title('Real dimwise distribution (x) vs Synthetic dimwise distribution (y)')
	plt.savefig('/home/gus/Documents/gan/medgan_maison/saved_outputs/last_output.png')
	plt.show()

def show(model):
	'''shortcut to display tf model architecture'''
	return tf.keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True)

def get_p_values(generator_trained_model, real_data, sample_size= 10000):
	'''gets p-values of a generator to compare its generated synthetic data with real data'''

	generated = simulate(generator_trained_model)
	sampled_real_data = tf.constant(pd.DataFrame(real_data.numpy()).sample(sample_size))

	ttest = ttest_ind(generated, sampled_real_data, axis=0)
	pvalues = ttest.pvalue
	return pvalues

def get_kl_norms(generator_trained_model, real_data, sample_size= 10000):
	'''gets kullback leibler norms for each dimension of the model'''

	generated = simulate(generator_trained_model)

	exp = tf.round(generated).numpy().astype(int)
	sampled_real_data = tf.constant(pd.DataFrame(real_data.numpy()).sample(sample_size))

	nmax= max(sampled_real_data.numpy().max(),exp.reshape(-1).max())

	Q = np.apply_along_axis(lambda dim: np.histogram(dim, bins= np.arange(nmax+1))[0], axis=0, arr=exp).T/sample_size
	P = np.apply_along_axis(lambda dim: np.histogram(dim, bins= np.arange(nmax+1))[0], axis=0, arr=sampled_real_data).T/sample_size

	Q[Q<10**-10] = 10**-10
	P[P<10**-10] = 10**-10

	kl_norm = np.dot(P, np.log(P/Q).T).diagonal()
	return kl_norm


def get_stats_save_models(g, d, ae, whole_data, cols, run_title='undefined'):
	'''compute every stat for a model, and save the dataframe of results'''

	s,r = moment_dim_wise(g, whole_data,mo=1)
	s2,r2 = moment_dim_wise(g, whole_data,mo=2)
	#pvals = get_p_values(g, whole_data)
	kln = get_kl_norms(g, whole_data)

	df = pd.DataFrame.from_dict(
	{'codes':cols, 'mo1_r':r, 'mo1_s':s, 'mo2_r':r2, 'mo2_s':s2, 
	#'pvals':pvals, 
	'klnorms':kln}
	).set_index('codes')

	#base_path = f'/home/gus/Documents/gan/results/{run_title}'
	base_path = f'C:/Users/I02COE1_A_COMBES0/Documents/medgan_2103/results_storage/{run_title}'
	os.mkdir(base_path)

	g.save(os.path.join(base_path,'generator'))
	d.save(os.path.join(base_path,'discriminator'))
	ae.save(os.path.join(base_path,'autoencoder'))

	df.to_csv(os.path.join(base_path, 'stats_dimwise.csv'),sep=';')