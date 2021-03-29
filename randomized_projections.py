from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import preprocessing

dataset = 2

X, y = preprocessing.preprocess(dataset)
X_train = StandardScaler().fit_transform(X)

X = range(2, 12)
d = {}
T = 10
for t in range(1, T):
    Y = []
    for n in X:
        model = GaussianRandomProjection(n_components=n, random_state=t)
        X_transformed = model.fit_transform(X_train)
        X_reconstructed = X_transformed.dot(np.linalg.pinv(model.components_.T))
        MSE = mean_squared_error(X_train, X_reconstructed)
        Y.append(MSE)
    d[t] = Y
df = pd.DataFrame.from_dict(d, orient='index')

if False:
    plt.xticks(X)
    plt.plot(X, Y)
    plt.title("RP, dataset " + str(dataset))
    plt.xlabel("Number of Components")
    plt.ylabel("MSE")
    plt.grid(b=True, which="major")
    plt.show()

if True:
    plt.xticks(X)
    plt.plot(X, df.mean())
    plt.errorbar(X, df.mean(), yerr=df.std())
    plt.title("RP, dataset " + str(dataset) + ", trials: " + str(T))
    plt.xlabel("Number of Components")
    plt.ylabel("MSE")
    plt.grid(b=True, which="major")
    plt.show()



