import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

import preprocessing
dataset = 2

X_train, y_train, X_test, y_test = preprocessing.preprocess(dataset)

model = KMeans(init='k-means++')
visualizer = KElbowVisualizer(model, k=(2, 15), metric="distortion")
o = visualizer.fit(X_train)
print("ok")


# visualizer.show(outpath="figures/kmeans/{0}_silhouette.png".format(dataset))