# sca_GAN
##### GAN mimicking EHR health data to realize more ethical predictions on sudden cardiac arrest in Paris' area population

## Overview of the problem
The objective of this work is to mimick healthcare consumption of real sudden-cardiac-arrest patients from Paris and its suburbs. The data used was provided by the CEMS (sudden cardiac arrest expert center) at Georges Pompidou European Hospital, in collaboration with INSERM, AP-HP and the Paris-Cité University. 

This two-in-one work mimicks both tabular data (see medgan_tabular model) and sequential data (see medgan_sequential model) : while in both cases, the model will create synthetic patient data, the two models differ in the data format. 
- Tabular data refers to the count of occurence of every int-encoded ehr event available: for instance, when creating a synthetic patient with 100 possible EHR events, it will generate a vector in |N^100, that represents the healthcare consumption of the synthetic patient over 5 years.
- Sequential data refers to generating a series of week-wise events: in this case, when creating a synthetic patient with 100 possible EHR events, it will generate n_timestep vectors of week-restricted tabular data in |N^100, each of which represents the healthcare consumption of the synthetic patient during a particular week, over 5 years.

While the work was initially designed for healthcare, the models can actually inspire ideas for other applications, when it does not solve the problem outright.

## Models used
Each method takes profit of deep learning to generate synthetic data. The main idea is to reduce the problem's dimension using a Variational AutoEncoder (VAE) and create a latent space to embed medical events from a discrete, high-dimensional distribution to a real, lower-dimensional one. Then, a Generative Adversarial Network (GAN) uses this latent space, more specifically the Decoder of the VAE, to both generate synthetic patients and discriminate them from real ones, in the usual adversarial loop for GANs.

In the case of the tabular model, both the Generator and the Discriminator are based on Dense (e.g fully-connected) layers, while in the sequential one, they are based on LSTM (Long Short-Term Memory) layers.

## Reference
Each of these methods are inspired by the following article :
  Generating Multi-label Discrete Patient Records using Generative Adversarial Networks
  Edward Choi, Siddharth Biswal, Bradley Malin, Jon Duke, Walter F. Stewart, Jimeng Sun  
  Machine Learning for Healthcare (MLHC) 2017
