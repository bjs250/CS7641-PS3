import preprocessing

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn import mixture
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn import cluster



from yellowbrick.cluster import KElbowVisualizer
import collections

def kurt(x):
    n = np.shape(x)[0]
    mean = np.sum((x**1)/n) # Calculate the mean
    var = np.sum((x-mean)**2)/n # Calculate the variance
    skew = np.sum((x-mean)**3)/n # Calculate the skewness
    kurt = np.sum((x-mean)**4)/n # Calculate the kurtosis
    kurt = np.abs(kurt/(var**2)-3)

    return kurt, skew, var, mean

dataset = 2

X, y = preprocessing.preprocess(dataset)
X = StandardScaler().fit_transform(X)

pca = PCA(n_components=0.95)
pca.fit(X)
X_pca_transformed = pca.transform(X)

# Determine k for Kmeans
if False:
    model = KMeans(init='k-means++')
    visualizer = KElbowVisualizer(model, k=(2, 15), metric="distortion")
    o = visualizer.fit(X_pca_transformed)

    visualizer.show(outpath="figures/kmeans/part3_{0}_distortion.png".format(dataset))

# Kmeans PCA
if False:
    if dataset == 1:
        n = 8
    if dataset == 2:
        n = 7
    model = KMeans(init='k-means++', n_clusters=n)
    model.fit(X_pca_transformed)
    y_pred = model.predict(X_pca_transformed)
    dist = collections.Counter(y_pred)
    dist_norm = [v / y_pred.size for v in dist.values()]
    dist_norm.sort()
    for v in dist_norm:
        print(v)
    print(homogeneity_completeness_v_measure(y.iloc[:, 1], y_pred))

# determine k for EM
if False:
    n_components = np.arange(1, 10)
    models = [mixture.GaussianMixture(n, covariance_type='full', random_state=0).fit(X_pca_transformed)
              for n in n_components]

    plt.plot(n_components, [m.bic(X_pca_transformed) for m in models], label='BIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.ylabel('BIC')
    plt.title('EM, dataset ' + str(dataset))
    plt.axvline(x=5, color='r', linestyle='-')
    plt.xticks(n_components)
    plt.grid()
    plt.show()

# EM PCA
if False:
    if dataset == 1:
        n = 8
    if dataset == 2:
        n = 7
    model = mixture.GaussianMixture(n, covariance_type='full', random_state=0)
    model.fit(X_pca_transformed)
    y_pred = model.predict(X_pca_transformed)
    dist = collections.Counter(y_pred)
    dist_norm = [v / y_pred.size for v in dist.values()]
    dist_norm.sort()
    for v in dist_norm:
        print(v)
    print(homogeneity_completeness_v_measure(y.iloc[:, 1], y_pred))

# run ICA
if dataset == 1:
    m = 12
if dataset == 2:
    m = 8
ica = FastICA(n_components=m)
X_ica_transformed = ica.fit_transform(X)  # Reconstruct signals

# Determine k for Kmeans
if False:
    model = KMeans(init='k-means++')
    visualizer = KElbowVisualizer(model, k=(2, 15), metric="distortion")
    o = visualizer.fit(X_ica_transformed)

    visualizer.show(outpath="figures/kmeans/part3_ica_{0}_distortion.png".format(dataset))

# Kmeans ICA
if False:
    if dataset == 1:
        n = 7
    if dataset == 2:
        n = 7
    model = KMeans(init='k-means++', n_clusters=n)
    model.fit(X_ica_transformed)
    y_pred = model.predict(X_ica_transformed)
    dist = collections.Counter(y_pred)
    dist_norm = [v / y_pred.size for v in dist.values()]
    dist_norm.sort()
    for v in dist_norm:
        print(v)
    print(homogeneity_completeness_v_measure(y.iloc[:, 1], y_pred))

# Determine k for EM
if False:
    n_components = np.arange(1, 10)
    models = [mixture.GaussianMixture(n, covariance_type='full', random_state=0).fit(X_ica_transformed)
              for n in n_components]

    plt.plot(n_components, [m.bic(X_ica_transformed) for m in models], label='BIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.ylabel('BIC')
    plt.title('EM, dataset ' + str(dataset))
    plt.axvline(x=5, color='r', linestyle='-')
    plt.xticks(n_components)
    plt.grid()
    plt.show()

# EM ICA
if False:
    if dataset == 1:
        n = 9
    if dataset == 2:
        n = 6
    model = mixture.GaussianMixture(n, covariance_type='full', random_state=0)
    model.fit(X_ica_transformed)
    y_pred = model.predict(X_ica_transformed)
    dist = collections.Counter(y_pred)
    dist_norm = [v / y_pred.size for v in dist.values()]
    dist_norm.sort()
    for v in dist_norm:
        print(v)
    print(homogeneity_completeness_v_measure(y.iloc[:, 1], y_pred))

if dataset == 1:
    p = 14
if dataset == 2:
    p = 10
model = GaussianRandomProjection(n_components=p, random_state=0)
X_rp_transformed = model.fit_transform(X)

# Determine k for Kmeans
if False:
    model = KMeans(init='k-means++')
    visualizer = KElbowVisualizer(model, k=(2, 15), metric="distortion")
    o = visualizer.fit(X_rp_transformed)

    visualizer.show(outpath="figures/kmeans/part3_rp_{0}_distortion.png".format(dataset))

# kmeans RP
if False:
    if dataset == 1:
        n = 6
    if dataset == 2:
        n = 5
    model = KMeans(init='k-means++', n_clusters=n)
    model.fit(X_rp_transformed)
    y_pred = model.predict(X_rp_transformed)
    dist = collections.Counter(y_pred)
    dist_norm = [v / y_pred.size for v in dist.values()]
    dist_norm.sort()
    for v in dist_norm:
        print(v)
    print(homogeneity_completeness_v_measure(y.iloc[:, 1], y_pred))

# Determine k for EM
if False:
    n_components = np.arange(1, 10)
    models = [mixture.GaussianMixture(n, covariance_type='full', random_state=0).fit(X_rp_transformed)
              for n in n_components]

    plt.plot(n_components, [m.bic(X_rp_transformed) for m in models], label='BIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.ylabel('BIC')
    plt.title('EM, dataset ' + str(dataset))
    plt.axvline(x=5, color='r', linestyle='-')
    plt.xticks(n_components)
    plt.grid()
    plt.show()

# EM RP
if False:
    if dataset == 1:
        n = 7
    if dataset == 2:
        n = 4
    model = mixture.GaussianMixture(n, covariance_type='full', random_state=0)
    model.fit(X_rp_transformed)
    y_pred = model.predict(X_rp_transformed)
    dist = collections.Counter(y_pred)
    dist_norm = [v / y_pred.size for v in dist.values()]
    dist_norm.sort()
    for v in dist_norm:
        print(v)
    print(homogeneity_completeness_v_measure(y.iloc[:, 1], y_pred))

if dataset == 1:
    q = 3
if dataset == 2:
    q = 5
model = cluster.FeatureAgglomeration(n_clusters=q)
model.fit(X)
X_fa_transformed = model.transform(X)

# Determine k for FA
if False:
    model = KMeans(init='k-means++')
    visualizer = KElbowVisualizer(model, k=(2, 15), metric="distortion")
    o = visualizer.fit(X_fa_transformed)

    visualizer.show(outpath="figures/kmeans/part3_fa_{0}_distortion.png".format(dataset))

# PCA FA
if False:
    if dataset == 1:
        n = 5
    if dataset == 2:
        n = 6
    model = KMeans(init='k-means++', n_clusters=n)
    model.fit(X_fa_transformed)
    y_pred = model.predict(X_fa_transformed)
    dist = collections.Counter(y_pred)
    dist_norm = [v / y_pred.size for v in dist.values()]
    dist_norm.sort()
    for v in dist_norm:
        print(v)
    print(homogeneity_completeness_v_measure(y.iloc[:, 1], y_pred))

# Determine k for EM
if False:
    n_components = np.arange(1, 10)
    models = [mixture.GaussianMixture(n, covariance_type='full', random_state=0).fit(X_fa_transformed)
              for n in n_components]

    plt.plot(n_components, [m.bic(X_fa_transformed) for m in models], label='BIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.ylabel('BIC')
    plt.title('EM, dataset ' + str(dataset))
    plt.axvline(x=5, color='r', linestyle='-')
    plt.xticks(n_components)
    plt.grid()
    plt.show()

# EM RP
if True:
    if dataset == 1:
        n = 4
    if dataset == 2:
        n = 5
    model = mixture.GaussianMixture(n, covariance_type='full', random_state=0)
    model.fit(X_fa_transformed)
    y_pred = model.predict(X_fa_transformed)
    dist = collections.Counter(y_pred)
    dist_norm = [v / y_pred.size for v in dist.values()]
    dist_norm.sort()
    for v in dist_norm:
        print(v)
    print(homogeneity_completeness_v_measure(y.iloc[:, 1], y_pred))