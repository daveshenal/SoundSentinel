import librosa
import numpy as np

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

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=22050, n_steps=pitch_factor)