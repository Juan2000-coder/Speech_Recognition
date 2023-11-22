import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import pdist, cdist
from mpl_toolkits.mplot3d import Axes3D
import soundfile as sf
from prettytable import PrettyTable
from scipy.signal import hilbert
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.signal import hilbert, butter, filtfilt

fruit_types = ['pera', 'banana', 'manzana', 'naranja']
fruits = {fruit: [] for fruit in fruit_types}
root_dir = './dataset'

for dirname, _, filenames in os.walk(root_dir):
    fruit_type = os.path.basename(dirname)
    if fruit_type in fruit_types:
        fruits[fruit_type].extend([os.path.join(dirname, filename) for filename in filenames if filename.endswith('.wav')])
trimed_audio = {fruit: [] for fruit in fruit_types}
for dirname, _, filenames in os.walk(root_dir):
    trimpath = os.path.basename(dirname)

    if trimpath in 'trimed':
        fruit_type = os.path.basename(os.path.dirname(dirname))
        if fruit_type in fruit_types:
            trimed_audio[fruit_type].extend([os.path.join(dirname, filename) for filename in filenames if filename.endswith('.wav')])
adjusted_audio = {fruit: [] for fruit in fruit_types}
for dirname, _, filenames in os.walk(root_dir):
    adjustedpath = os.path.basename(dirname)

    if adjustedpath in 'adjusted':
        fruit_type = os.path.basename(os.path.dirname(dirname))
        if fruit_type in fruit_types:
            adjusted_audio[fruit_type].extend([os.path.join(dirname, filename) for filename in filenames if filename.endswith('.wav')])

FRAME_SIZE = 512 # In the documentation says it's convenient for speech.C
HOP_SIZE   = 256

def load_audio(audiofile):
    test_audio, sr = librosa.load(audiofile, sr = None)
    duration = librosa.get_duration(filename=audiofile, sr=sr)
    return test_audio, sr, duration
def calculate_smoothed_envelope(signal, sr, cutoff_frequency = 10.0):
    analytic_signal = hilbert(signal)

    # Calcular la envolvente de amplitud
    amplitude_envelope = np.abs(analytic_signal)

    # Aplicar filtro pasa bajos para suavizar la envolvente
    nyquist = 0.5 * sr
    cutoff = cutoff_frequency / nyquist
    b, a = butter(N=6, Wn=cutoff, btype='low', analog=False, output='ba')
    
    smoothed_envelope = filtfilt(b, a, amplitude_envelope)

    return amplitude_envelope, smoothed_envelope
def get_features(n_mfcc, fruits):
    fruit_vectors = dict.fromkeys(fruits.keys())
    
    for fruit_name, group in fruits.items():
        vectors = list()
        for fruit in group:
            signal, sr, _ = load_audio(fruit)

            mfccs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, sr=sr)
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order = 2)
            
            _, smoothed_envelope = calculate_smoothed_envelope(signal, sr, 45)

            selected_indices = np.linspace(0, len(smoothed_envelope) - 1, n_mfcc, dtype=int)
            smoothed_envelope = smoothed_envelope[selected_indices]
            smoothed_envelope = smoothed_envelope.reshape(-1,1)

            smoothed_envelope = smoothed_envelope /np.max(np.linalg.norm(smoothed_envelope , axis=0))
            mfccs = mfccs/np.max(np.linalg.norm(mfccs, axis=0))
            delta_mfccs  = delta_mfccs/np.max(np.linalg.norm(delta_mfccs, axis=0))
            delta2_mfccs = delta2_mfccs/np.max(np.linalg.norm(delta2_mfccs, axis=0))

            mfccs = np.mean(mfccs.T, axis = 0)
            mfccs = mfccs.reshape(-1, 1)
            delta_mfccs = np.mean(delta_mfccs.T, axis = 0)
            delta_mfccs = mfccs.reshape(-1, 1)
            delta2_mfccs = np.mean(delta2_mfccs.T, axis = 0)
            delta2_mfccs = mfccs.reshape(-1, 1)

            #features =  np.concatenate((delta_mfccs), axis = 0)
            features = smoothed_envelope
            #features = np.mean(features.T, axis = 0)
            vectors.append(features.reshape(1,-1))
            
        fruit_vectors[fruit_name] = np.vstack(vectors)
    return fruit_vectors

def get_sphere(vectors):
    center = np.mean(vectors, axis = 0)
    center = center.reshape(1, -1)
    radius = cdist(center, vectors).max()     # Pairwise distance
    return radius, center
def get_centers(features):
    centers = dict.fromkeys(features.keys())
    for fruit, group in features.items():
        _, center = get_sphere(group)
        centers[fruit] = center
    return centers
def get_radiuses(features):
    radiuses = dict.fromkeys(features.keys())
    for fruit, group in features.items():
        radius, _ = get_sphere(group)
        radiuses[fruit] = radius
    return radiuses
def get_overlaps(fruit_features):
    centers = get_centers(fruit_features)
    radiuses = get_radiuses(fruit_features)
    overlaps = dict.fromkeys(fruit_features.keys())
    
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
    return overlaps # Each element is the number of vectors of one group in the sphere of another
def get_components(centers, nc):
    pacum = np.zeros((1, centers[fruit_types[0]].shape[1]))
    pair_components = dict()
    for i in range(len(fruit_types)):
        for j in range(i + 1, len(fruit_types)):
            dif = centers[fruit_types[i]] - centers[fruit_types[j]]
            dist = cdist(centers[fruit_types[i]], centers[fruit_types[j]])
            difp = (dif**2)*100/(dist**2)
            pair_components[f"{fruit_types[i]}-{fruit_types[j]}"]= np.argsort(difp[0])[-nc:]
            pacum += difp
    index_max = np.argsort(pacum[0])[-nc:]
    #return np.sort(index_max)
    return index_max, pair_components

features = get_features(30, trimed_audio)
overlaps = get_overlaps(features)
centers = get_centers(features)
components, _ = get_components(centers, 2) 
radius, _ = get_sphere(np.squeeze(list(centers.values()), axis=1))

print(f"componentes: {components}")

for pair, overlap in overlaps.items():
    print(f"{pair}: {overlap}")

print(f"radius: {radius}")

for name, group in features.items():
    features[name] = group[:, np.sort(components)]
"""
whole = np.concatenate(list(features.values()), axis=0)

#Paso 2: Aplicar PCA para obtener dos componentes principales
pca = PCA(n_components = 3)
reduced_features = pca.fit_transform(whole)

#Paso 3: Crear un diccionario con las matrices reducidas
reduced = {}
start_idx = 0

for fruit, matrix in features.items():
    num_rows = matrix.shape[0]
    reduced[fruit] = reduced_features[start_idx:start_idx + num_rows, :]
    start_idx += num_rows
features = reduced
"""
def knn(training, test, k_n):
    X = np.concatenate([v for v in training.values()], axis=0)
    y = np.concatenate([[k] * v.shape[0] for k, v in training.items()])

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear clasificador KNN
    knn_classifier = KNeighborsClassifier(n_neighbors = k_n)

    # Entrenar el clasificador
    knn_classifier.fit(X, y)

    # Predecir las etiquetas para los datos de prueba
    predicted_fruit = knn_classifier.predict(test)

    print(f'La fruta predicha para el nuevo audio es: {predicted_fruit[0]}')

#3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = dict(zip(fruit_types,['green','yellow','red','orange']))
for fruit, points in features.items():
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors[fruit], marker='o', label=fruit)
    #ax.scatter(centers[fruit][:, 0], centers[fruit][:, 1], centers[fruit][:, 2], c=center_colors[fruit], marker='o', label=f"{fruit}-center")

# configure labels
ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')
ax.set_zlabel('Eje Z')
plt.show()