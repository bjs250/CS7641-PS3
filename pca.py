from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

import preprocessing

dataset = 2

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


X_train, y_train, X_test, y_test = preprocessing.preprocess(dataset)
X_train = StandardScaler().fit_transform(X_train)
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