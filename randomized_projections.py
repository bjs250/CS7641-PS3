from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

import preprocessing
dataset = 2

X, y = preprocessing.preprocess(dataset)
X_train = StandardScaler().fit_transform(X)

X = range(2, 13)
Y = []
for n in X:
    model = GaussianRandomProjection(n_components=n)
    X_transformed = model.fit_transform(X_train)
    X_reconstructed = X_transformed.dot(np.linalg.pinv(model.components_.T))
    MSE = mean_squared_error(X_train, X_reconstructed)
    Y.append(MSE)

plt.xticks(X)
plt.plot(X, Y)
plt.title("RP, dataset " + str(dataset))
plt.xlabel("Number of Components")
plt.ylabel("MSE")
plt.grid(b=True, which="major")
plt.show()


