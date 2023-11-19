# Imports
import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import pdist, cdist

# Diccionary with the list of fruits
fruit_types = ['pera', 'banana', 'manzana', 'naranja']
fruit_dict = {fruit: [] for fruit in fruit_types}
root_dir = '../../dataset'

for dirname, _, filenames in os.walk(root_dir):
    fruit_type = os.path.basename(dirname)
    if fruit_type in fruit_types:
        fruit_dict[fruit_type].extend([os.path.join(dirname, filename) for filename in filenames if filename.endswith('.wav')])

# Audio parameters
FRAME_SIZE = 512 # In the documentation says it's convenient for speech.C
HOP_SIZE   = 256

# Function to load the audio signal
def load_audio(audiofile):
    test_audio, sr = librosa.load(audiofile, sr = None)
    duration = len(test_audio)/sr
    return test_audio, sr, duration

# Function to plot the signal
def plot_signal(signal, sample_rate):
    # Duration and time vector
    duration = len(signal)/sample_rate
    
    # Time_vector
    time_vector = np.linspace(0, duration, len(signal))
    
    # Figure
    plt.figure(figsize=(15, 5))
    
    #plot
    librosa.display.waveshow(signal, sr=sample_rate, axis='time')
    
    #plt.plot(time_vector, signal, linestyle='-')
    
    # extra
    plt.title(f'Señal de Audio - Duración: {duration:.2f} segundos')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.show()

# Función de ploteo del espectrograma
def plot_spectrogram(stft_power, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(stft_power, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    plt.show()

# Low pass filter
def low_pass_filter(signal, cutoff_frequency, sample_rate):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Function to get the RMS values
def get_rms(signal):
    return librosa.feature.rms(y=signal, frame_length = FRAME_SIZE, hop_length = HOP_SIZE)

# Function to trim the audio signal
def trim(signal, perc):             # The audio signal and the percentaje umbral
    signal_rms = get_rms(signal)   # RMS values for each fram
    max_rms = signal_rms.max()      # Max RMS

    umbral = perc*max_rms/100       # Umbral

    mask_rms = signal_rms > umbral

    rms_bottom_index = np.argmax(mask_rms)
    rms_upper_index = mask_rms.shape[1] - np.argmax(np.flip(mask_rms))

    bottom_index = HOP_SIZE*rms_bottom_index
    upper_index  = HOP_SIZE*(rms_upper_index + 1)

    return signal[bottom_index:upper_index]   # Trimmed signal

# Gets a diccionary with the names of the fruits a key and a list of vectors(features) as values
def get_features(n_mfcc, components:list):
    fruit_vectors = dict.fromkeys(fruit_types)
    for fruit_name, fruits in fruit_dict.items():
        vectors = list()
        for fruit in fruits:
            signal, sr, _ = load_audio(fruit)
            signal = trim(signal, 15)

            #signal_stft  = librosa.stft(signal, n_fft = FRAME_SIZE, hop_length = HOP_SIZE) # STFT
            #signal_power = np.abs(signal_stft) ** 2                                        # Power Spectrum
            #signal_power_db = librosa.power_to_db(signal_power)                            # Simply gets the db values
            #signal_mel_spectrogram = librosa.feature.melspectrogram(y = signal, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, n_mels=20) # Mel spectrogram
            #signal_mel_spectrogram_db = librosa.power_to_db(signal_mel_spectrogram)

            mfccs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, sr=sr)                      # Gets the Mel coeeficients
            fruit_vector = np.mean(mfccs, axis=1)
            fruit_vector = fruit_vector[components]
            vectors.append(fruit_vector.T)
        fruit_vectors[fruit_name] = np.array(vectors)

    return fruit_vectors

# See how far are the freatures with each other in each group. We are gonna define the diameter
def get_sphere(vectors):
    center = np.mean(vectors, axis = 0)
    center = center.reshape((1, len(center)))
    radius = cdist(center, vectors).max()     # Pairwise distance
    return radius, center

# Function to calculate the overlaps
def get_overlaps(fruit_features):
    radiuses = dict.fromkeys(fruit_types) # A dictionary for the radius
    centers = dict.fromkeys(fruit_types)  # A dictionary for the centers

    for name, group in fruit_features.items():
        radius, center = get_sphere(group) # The sphere that evolves the group
        radiuses[name] = radius
        centers[name] = center

    overlaps = dict.fromkeys(fruit_types)
    # A dictionary of dictionarys. Keys, the fruit types
    for key in overlaps:
        # Each dictionary in the dictionary
        overlaps[key] = dict.fromkeys(fruit_types)
    
    for i in range(len(fruit_types)):
        for j in range(i + 1, len(fruit_types)):
            distancesAB = cdist(centers[fruit_types[i]], fruit_features[fruit_types[j]])
            distancesBA = cdist(centers[fruit_types[j]], fruit_features[fruit_types[i]])

            mask_distancesAB = distancesAB < radiuses[fruit_types[i]]
            mask_distancesBA = distancesBA < radiuses[fruit_types[j]]

            numberBinA = np.count_nonzero(mask_distancesAB)
            numberAinB = np.count_nonzero(mask_distancesBA)

            overlaps[fruit_types[i]][fruit_types[j]] = numberBinA
            overlaps[fruit_types[j]][fruit_types[i]] = numberAinB
    return overlaps # Each element is the number of vectors of one group in the sphere of anhoter

# Gets an informative table with the distance between centers
def get_table(fruit_features):
    centers = dict.fromkeys(fruit_types)  # A dictionary for the centers

    for name, group in fruit_features.items(): # Get the centers
        _, center = get_sphere(group)     # The sphere that evolves the group
        centers[name] = center

    from prettytable import PrettyTable
    table = PrettyTable()
    table.field_names = ["par"] + [f"{i}" for i in range(1, centers[fruit_types[0]].shape[1] + 1)]

    for i in range(len(fruit_types)):
        for j in range(i + 1, len(fruit_types)):

            key = f"{fruit_types[i]}-{fruit_types[j]}"
            centerdif = centers[fruit_types[i]] - centers[fruit_types[j]]
            difperc = centerdif*100/cdist(centers[fruit_types[i]], centers[fruit_types[j]])

            table.add_row([key] + centerdif[0].tolist())
            table.add_row([f"{key}-%"] + difperc[0].tolist())
    table.float_format = "0.2"
    return table

import mplcursors
fruit_features = get_features(10, [0, 1, 4])
#print(get_table(fruit_features))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Configurar colores y marcas para cada grupo

colors = dict(zip(fruit_types,['green','yellow','red','orange']))

# Plotear los puntos para cada grupo
for fruit, puntos in fruit_features.items():
    ax.scatter(puntos[:, 0], puntos[:, 1], puntos[:, 2], c=colors[fruit], marker='o', label=fruit)

# Configurar etiquetas de ejes
ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')
ax.set_zlabel('Eje Z')

# Configurar leyenda
ax.legend()

# Agregar cursores interactivos
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f'({sel.target[0]:.2f}, {sel.target[1]:.2f})'))
plt.show()