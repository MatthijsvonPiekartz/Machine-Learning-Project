# Matthijs von Piekartz SKEPP 10-04-2019
# Libraries importeren

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
from sklearn.cluster import KMeans
import matplotlib.cm
cmap = matplotlib.cm.get_cmap('plasma')


ds = pd.read_csv('training_set2.csv')
X = ds.iloc[:, [3,4]].values

# Waarde van k kiezen met de elbow method (Hoeveelheid clusters kiezen)
wcss = []

for i in range(1,21):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit_transform(X)
    wcss.append(kmeans.inertia_)
    
plt.figure()
plt.plot(range(1,21), wcss)
plt.title('Elbow Methode')
plt.xlabel('Aantal clusters')
plt.ylabel('Verschil tussen de observaties')
plt.show()

# Data clusteren
k = 5
kmeans = KMeans(n_clusters = k)
y_kmeans = kmeans.fit_predict(X)

labels = [('Cluster ' + str(i+1)) for i in range(k)]

# Clusters plotten
plt.figure()
for i in range(k):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s = 20,
                 c = cmap(i/k), label = labels[i]) 
 
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            s = 100, c = 'black', label = 'Centroids', marker = 'X')
plt.xlabel('Leeftijd')
plt.ylabel('Budget')
plt.title('K-means Clustering SKEPP Test')
plt.legend()
plt.show()
    