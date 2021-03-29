import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

dataset = 2

X, y = preprocessing.preprocess(dataset)
X_train = StandardScaler().fit_transform(X)

Y = []
N = range(2, 10)
for n in N:
    model = cluster.FeatureAgglomeration(n_clusters=n)
    model.fit(X)
    X_reduced = model.transform(X)
    X_reconstructed = model.inverse_transform(X_reduced)
    MSE = mean_squared_error(X, X_reconstructed)
    Y.append(MSE)

if True:
    plt.xticks(N)
    plt.plot(N, Y)
    plt.title("FA, dataset " + str(dataset))
    plt.xlabel("Number of Components")
    plt.ylabel("MSE")
    plt.grid(b=True, which="major")
    plt.show()