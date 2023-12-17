import joblib
import matplotlib.pylab as plt
import os
import numpy as np

features_dict = joblib.load('./implementation/images/kmeans_tests/features.pkl')

#3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = dict(zip(features_dict.keys(), ['yellow','green','orange','red']))

for key, cluster in features_dict.items():
    ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], c=colors[key], marker='o', label=key)
        
ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')
ax.set_zlabel('Eje Z')
plt.show()