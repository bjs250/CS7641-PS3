import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import homogeneity_completeness_v_measure
import collections

import numpy as np
import preprocessing

dataset = 1

X, y = preprocessing.preprocess(dataset)
X = StandardScaler().fit_transform(X)

# Generate plot
if False:
    n_components = np.arange(1, 10)
    models = [mixture.GaussianMixture(n, covariance_type='full', random_state=0).fit(X)
              for n in n_components]

    plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.ylabel('BIC')
    plt.title('EM, dataset ' + str(dataset))
    plt.axvline(x=5, color='r', linestyle='-')
    plt.xticks(n_components)
    plt.grid()
    plt.show()

if True:
    n = 6
    model = mixture.GaussianMixture(n, covariance_type='full', random_state=0)
    model.fit(X)
    y_pred = model.predict(X)
    dist = collections.Counter(y_pred)
    for v in dist.values():
        print(v/y_pred.size)
    print(homogeneity_completeness_v_measure(y.iloc[:, 1], y_pred))