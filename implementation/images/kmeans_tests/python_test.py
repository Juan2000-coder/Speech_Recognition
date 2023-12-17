import os
import numpy as np
import cv2
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
import joblib
from scipy.spatial.distance import cdist

def kmeans(n_clusters, features, tol = 1e-4, max_iter = 300):
    centroids   = []
    range_min   = np.min(features, axis = 0)  # El mínimo de las componentes de los vectores
    range_max   = np.max(features, axis = 0)  # El máximo de las componentes de los vectores

    centroids = np.random.uniform(range_min, range_max, size = (n_clusters, features.shape[1]))
    # Obtenemos los centroides iniciales de forma aleatoria
    '''for _ in range(n_clusters):
        centroid = np.random.uniform(range_min, range_max, size = (3,))
        centroids.append(centroid)'''

    # Los colocamos a los centroiddes como filas de una matriz
    centroids = np.vstack(centroids)

    for _ in range(max_iter):
        # Clusters
        clusters        = []
        labels          = np.empty(features.shape[0])
        new_centroids   = []
        
        # Distancias de cada punto a cada centroide
        dist        = cdist(centroids, features)
        sorted_ind  = np.argsort(dist, axis = 0)

        # Construcción de los clusters
        # Y recalculo de los centroides
        
        for j in range(n_clusters):
            index         = sorted_ind[0, :] == j
            cluster       = features[index, :]
            labels[index] = j
            clusters.append(cluster)
            centroid      = np.mean(cluster, axis = 0)
            new_centroids.append(centroid)
        new_centroids = np.vstack(new_centroids)

        # Verificamos condicion de parada por tolerancia
        dist = np.linalg.norm(centroids - new_centroids, axis = 1)
        if np.max(dist) < tol:
            break

        centroids = new_centroids
    # Devolvemos la matriz que tiene por filas los centroides.
        
    # Devolvemos la lisya que tiene por elementos las matrices que tienen por filas
    # los puntos del cluster.
    
    # Devolvemos la lista de labels que indican la pertenencia a un cluster de cada elemento.
    return  list(labels), clusters, centroids

features = joblib.load('./implementation/images/kmeans_tests/points.pkl')

# Especificar el número de clusters (k)
num_clusters = 4

# Aplicamos kmeans
labels,_,_ = kmeans(num_clusters, features)