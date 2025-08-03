import librosa
import numpy as np
from src.config.config_loader import load_config

config = load_config("src/config/config.yaml")

sample_rate = config["audio"]["sample_rate"]
pitch_factor = config["audio"]["pitch_factor"]
stretch_rate = config["audio"]["stretch_rate"]

'''
To generate syntactic data for audio, we can apply noise injection, shifting time,
changing pitch and speed.

They help introduce variability in the training data, making the model more robust
by simulating real-world variations in sound.

noise   : adds random noise to the audio signal to simulate background noise.
stretch : stretches or compresses the audio in time without affecting its pitch.
shift   : shifts the audio signal in time, creating variations in the starting 
          point of the audio.
pitch   : This method shifts the pitch of the audio, simulating changes in the
          speaker's tone or pitch.

'''

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data):
    return librosa.effects.time_stretch(data, rate=stretch_rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data):
    return librosa.effects.pitch_shift(data, sr=sample_rate, n_steps=pitch_factor)