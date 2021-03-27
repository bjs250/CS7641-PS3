import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.preprocessing import StandardScaler
import collections

import preprocessing
dataset = 1

X, y = preprocessing.preprocess(dataset)
X_train = StandardScaler().fit_transform(X)

if False:
    model = KMeans(init='k-means++')
    visualizer = KElbowVisualizer(model, k=(2, 15), metric="distortion")
    o = visualizer.fit(X)

    visualizer.show(outpath="figures/kmeans/{0}_distortion.png".format(dataset))

if False:
    N = range(2, 15)
    v_scores = []
    for n in N:
        model = KMeans(init='k-means++', n_clusters=n)
        model.fit(X)
        y_pred = model.predict(X)
        v_scores.append(v_measure_score(y.iloc[:,1], y_pred))

    plt.xticks(N)
    plt.bar(N, v_scores)
    plt.title("K-means clustering v_scores, dataset " + str(dataset))
    plt.xlabel("Number of Components")
    plt.ylabel("v-score")
    plt.grid(b=True, which="major")
    plt.show()

if True:
    n = 6
    model = KMeans(init='k-means++', n_clusters=n)
    model.fit(X)
    y_pred = model.predict(X)
    dist = collections.Counter(y_pred)
    for v in dist.values():
        print(v/y_pred.size)
    print(homogeneity_completeness_v_measure(y.iloc[:, 1], y_pred))


