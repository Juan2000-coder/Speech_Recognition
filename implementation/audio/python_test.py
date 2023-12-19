#**IMPORTS#**
import os
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import joblib

fruit_types      = ['pera', 'banana', 'manzana', 'naranja']
audios           = {fruit: [] for fruit in fruit_types}
training_path    = '../../../dataset/audios/training'
original_path    = os.path.join(training_path, 'original')
processed_path   = os.path.join(training_path, 'processed')
model_file       = './implementation/audio/knn/model.pkl'
model            = dict.fromkeys(['pca', 'features', 'scaler'])

#3d
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

model        = joblib.load(model_file)
reduced:dict = model['features']
pca          = model['pca']
scaler       = model['scaler']

plot_features3d(reduced)