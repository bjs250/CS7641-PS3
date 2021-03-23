from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import preprocessing

def kurt(x):
    n = np.shape(x)[0]
    mean = np.sum((x**1)/n) # Calculate the mean
    var = np.sum((x-mean)**2)/n # Calculate the variance
    skew = np.sum((x-mean)**3)/n # Calculate the skewness
    kurt = np.sum((x-mean)**4)/n # Calculate the kurtosis
    kurt = np.abs(kurt/(var**2)-3)

    return kurt, skew, var, mean

dataset = 1
X_train, y_train, X_test, y_test = preprocessing.preprocess(dataset)
X_train = StandardScaler().fit_transform(X_train)

print(kurt(X_train))

X = range(2, 15)
Y = []

for n in X:
    ica = FastICA(n_components=n)
    S_ = ica.fit_transform(X_train)  # Reconstruct signals
    Y.append(kurt(S_)[0])

plt.xticks(X)
plt.plot(X,Y)
plt.title("ICA, dataset " + str(dataset))
plt.xlabel("Number of Components")
plt.ylabel("|Kurtosis|")
plt.grid(b=True, which="major")
plt.show()


