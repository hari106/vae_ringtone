# vae_ringtone
## Generation of MIDI ringtones using VAE architecture

This project is a simple training experiment that trains a Variational AutoEncoder architecture on about 600 MIDI ringtones and generates new ones based on it. The reason why I chose VAE was because I had tried UNet, LSTM and RNN to get rather dismal results. VAE seems to retain a fraction of the musicality that can be seen in the samples. This is probably because VAE creates latent distributions from which we sample vectors instead of generaing vectors themselves, leading to capture of the overall "idea" of the ringtones, but still keeping it slightly original. 
