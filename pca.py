from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import preprocessing

dataset = 1

if dataset == 1:
    target_values = [0, 1]
    target_descriptors = ["=<50k, >50K"]
    n_components = 2
    target_column = 'result'
    colors = ['r', 'g', 'b']
elif dataset == 2:
    target_values = [False, True]
    # target_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    target_descriptors = ["<=6", ">6"]
    n_components = 2
    target_column = 'quality'
    colors = ['r', 'g', 'b']
    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'navy', 'crimson', 'deepskyblue']


X, y = preprocessing.preprocess(dataset)
X_train = StandardScaler().fit_transform(X)

if False:
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(X_train)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, y_train], axis = 1)

    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA: Dataset ' + str(dataset), fontsize = 20)
    for target, color in zip(target_values, colors):
        indicesToKeep = finalDf[target_column] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 10)
    ax.legend(target_descriptors)
    ax.grid()
    plt.show()

if False:
    pca = PCA(n_components=0.95)
    pca.fit(X_train)
    reduced = pca.transform(X_train)

    plt.rcParams["figure.figsize"] = (8, 8)

    fig, ax = plt.subplots()
    xi = np.arange(1, pca.explained_variance_ratio_.size + 1, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)

    plt.ylim(0.0, 1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, pca.explained_variance_ratio_.size + 1, step=1))  # change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('PCA, dataset ' + str(dataset))

    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=16)

    ax.grid()
    plt.show()

# Optimal
if True:
    mean_vec = np.mean(X, axis=0)
    cov_mat = (X - mean_vec).T.dot((X - mean_vec)) / (X.shape[0] - 1)
    print('Covariance matrix \n%s' % cov_mat)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    for i in range(1, eig_vals.size):
        print("{0}, {1}".format(cov_mat.columns[i], eig_vals[i]))


    pca = PCA(n_components=0.95)
    pca.fit(X_train)
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)