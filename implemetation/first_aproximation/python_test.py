import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

fruit_types = ['pera', 'banana', 'manzana', 'naranja']
FRAME_SIZE = 512 # In the documentation says it's convenient for speech.C
HOP_SIZE   = 256
# Directorios de los conjuntos de datos
root_dir = '.'

dir = dict.fromkeys(fruit_types)
for fruit in dir:
    dir[fruit] = os.path.join(root_dir,f"dataset/{fruit}/adjusted")

# Función para extraer características (espectrograma) de un archivo de audio
def get_feat(audio, n_mfcc):
    y, sr = librosa.load(audio, sr=None)
    stft = np.abs(librosa.stft(y, n_fft = FRAME_SIZE, hop_length = HOP_SIZE))
    mfccs = librosa.feature.mfcc(S = librosa.power_to_db(stft**2), n_mfcc = n_mfcc, sr=sr, n_fft = FRAME_SIZE, hop_length = HOP_SIZE)
    mfccs = np.mean(mfccs.T, axis = 0)
    return mfccs.reshape(1, -1)

# Leer y procesar los audios
n_mfcc = 13
features = dict.fromkeys(fruit_types)

etiquetas = []

for name in features:
    features[name] = [get_feat(os.path.join(dir[name], archivo), n_mfcc) for archivo in os.listdir(dir[name])]
    etiquetas += [name]*len(features[name])

# Concatenar las características y etiquetas
print(etiquetas)

whole = None
for fruit, group in features.items():
    features[fruit] = np.vstack(group)
    if whole is None:
        whole = features[fruit]
    else:
        whole = np.concatenate((whole, features[fruit]), axis = 0)

# Aplicar PCA para reducir la dimensionalidad
pca = PCA(n_components=2)
reduced = pca.fit_transform(whole)

for etiqueta in set(etiquetas):
    indices = np.where(np.array(etiquetas) == etiqueta)
    indices = sorted(indices[0])
    features[etiqueta] = reduced[indices,:]

plt.figure(figsize=(10, 8))
colors = dict(zip(fruit_types,['green','yellow','red','orange']))

print(features)

for fruit, points in features.items():
    plt.scatter(points[:, 0], points[:, 1], c = colors[fruit], label=fruit)
plt.scatter(20, 100, c='black')
# Graficar los puntos
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.title('Visualización de Características con PCA')
plt.show()
#plt.legend()
