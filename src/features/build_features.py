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

def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally

    return result

def get_features(path, label=None):
    # Load audio data without specifying duration and offset (they are fixed)
    data = librosa.load(path, sr=sample_rate)[0]

    # Extract features without augmentation
    res1 = extract_features(data)
    result = np.array(res1)

    # Apply augmentation only if the label is "glass_break"
    if label == "glass_break":
        # Add noise
        noise_data = noise(data)
        res2 = extract_features(noise_data)
        result = np.vstack((result, res2))

        # Stretch and pitch
        new_data = stretch(data)
        data_stretch_pitch = pitch(new_data)
        res3 = extract_features(data_stretch_pitch)
        result = np.vstack((result, res3))

    return result