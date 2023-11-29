#**IMPORTS#**
import os
import math
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
from scipy.signal import hilbert, butter, lfilter
import soundfile as sf
import sounddevice as sd
#**RUTAS Y TIPOS DE FRUTAS#**
fruit_types      = ['pera', 'banana', 'manzana', 'naranja']
dataset_path     = './dataset/audios'
original_path    = os.path.join(dataset_path, 'original')
processed_path   = os.path.join(dataset_path, 'processed')
o_tests_path     = os.path.join(original_path, 'tests')
p_tests_path     = os.path.join(processed_path, 'tests')
model_file       = './implemetation/knn/model.pkl'
model            = dict.fromkeys(['pca', 'features'])
#**ORIGINAL TESTS#**
o_tests = []
o_tests.extend([os.path.join(o_tests_path, filename) for filename in os.listdir(o_tests_path) if filename.endswith('.wav')])
#**PROCESSED TESTS DICT#**
p_tests = []
p_tests.extend([os.path.join(p_tests_path, filename) for filename in os.listdir(p_tests_path) if filename.endswith('.wav')])
#**PARAMETROS DEL AUDIO#**
FRAME_SIZE = 1024# In the documentation says it's convenient for speech.C
HOP_SIZE   = int(FRAME_SIZE/2)
#**FUNCIONES GENERALES DE AUDIO#**
def load_audio(audiofile):
    test_audio, sr = librosa.load(audiofile, sr = None)
    duration = librosa.get_duration(filename=audiofile, sr=sr)
    return test_audio, sr, duration
def time_vector(signal, duration):
    return np.linspace(0, duration, len(signal))
def rms(signal, frames, hop):
    return librosa.feature.rms(y=signal, frame_length = frames, hop_length = hop)
def normalize(signal):
    peak = np.max(signal)
    signal/=peak
    return signal
def derivative(signal, duration):
    signal = signal.reshape(-1,)
    dy = np.gradient(signal, np.linspace(0, duration, len(signal)))
    return dy
#**FILTERS#**
def low_pass_filter(signal, sr, cutoff_frequency = 5000):
    nyquist = 0.5 * sr
    cutoff = cutoff_frequency / nyquist
    b, a = butter(N=6, Wn=cutoff, btype='low', analog=False, output='ba')
    filtered = lfilter(b, a, signal)
    return filtered
def band_pass_filter(signal, sr, low_cutoff, high_cutoff):
    b, a = butter(N=3, Wn = [low_cutoff, high_cutoff], btype='band', fs=sr)
    return lfilter(b, a, signal)
def preemphasis(signal, coef=0.97):
    return np.append(signal[0], signal[1:] - coef * signal[:-1])
def envelope(signal):
    analytic_signal = hilbert(signal)
    return np.abs(analytic_signal)
def smooth_envelope(signal, sr, cutoff_frequency=50.0):
    return low_pass_filter(envelope(signal), sr, cutoff_frequency)
#**PROCCESSING OF THE AUDIO FILES FUNCTIONS#**
#*File naming*
def get_name(o_tests:list):
    return os.path.join(o_tests_path,"test" + f"{len(o_tests) + 1}" + ".wav")
#*Processing*
def process(audio_in, audio_out, umbral = 0.295):
    signal, sr, duration = load_audio(audio_in)
    
    filtered = low_pass_filter(signal, sr, 1800)
    filtered = preemphasis(filtered, 0.999)

    rms_signal = rms(signal, 4096, 2048)

    rms_signal = normalize(rms_signal)
    drms = normalize(derivative(rms_signal, duration))

    audio_vector = time_vector(signal, duration)
    drms_vector = time_vector(drms, duration)

    left_index = np.argmax(np.abs(drms) > umbral)
    rigth_index = len(drms) - 1 - np.argmax(np.abs(np.flip(drms)) > umbral)

    left_time = drms_vector[left_index]
    rigth_time = drms_vector[rigth_index]

    mask_vector = audio_vector >= left_time

    audio_vector = audio_vector[mask_vector]
    trimed_signal = signal[mask_vector]

    mask_vector = audio_vector <= rigth_time

    audio_vector = audio_vector[mask_vector]
    trimed_signal = trimed_signal[mask_vector]

    sf.write(audio_out, trimed_signal, sr)

def plot_features3d(features):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = dict(zip(fruit_types,['green','yellow','red','orange']))

    for fruit, points in features.items():
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors[fruit], marker='o', label=fruit)
        
    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    ax.set_zlabel('Eje Z')
    plt.show()
#3d
model = joblib.load(model_file)
reduced:dict = model['features']
pca = model['pca']
plot_features3d(reduced)