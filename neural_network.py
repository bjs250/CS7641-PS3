import preprocessing
import time

import pickle
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn import cluster

def get_best_parameters(X_train, y_train, path):

    parameters = {
        'hidden_layer_sizes': [(8), (8, 8, 8)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.01, 0.1, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }

    mlp = MLPClassifier(max_iter=1000)

    clf = GridSearchCV(
        mlp,
        parameters,
        cv=5,
        n_jobs=-1,
        return_train_score=True,
        verbose=10
    )
    clf.fit(X_train, y_train)

    best_params = clf.best_params_
    file_pi = open(path, 'wb')
    pickle.dump(best_params, file_pi)

    return best_params

def evaluate(X_train, y_train, X_test, y_test, parameters):

    mlp = MLPClassifier(
        hidden_layer_sizes=parameters['hidden_layer_sizes'],
        activation=parameters['activation'],
        solver=parameters['solver'],
        alpha=parameters['alpha'],
        learning_rate=parameters['learning_rate'],
        max_iter=1500
    )
    start = time.time()
    mlp.fit(X_train, y_train.values.ravel())
    stop = time.time()
    print(f"Training time: {stop - start}s")
    y_pred_train = mlp.predict(X_train)
    y_pred_test = mlp.predict(X_test)

    train_acc  = metrics.accuracy_score(y_train, y_pred_train)
    test_acc = metrics.accuracy_score(y_test, y_pred_test)
    print("Train Accuracy: ", train_acc)
    print("Test Accuracy", test_acc)

    print(precision_recall_fscore_support(y_test, y_pred_test, average='weighted'))
    print(confusion_matrix(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test))

def pca(X):
    pca = PCA(n_components=0.95)
    pca.fit(X)
    return pca.transform(X)

def ica(X, m):
    ica = FastICA(n_components=m)
    return ica.fit_transform(X)

def rp(X, m):
    model = GaussianRandomProjection(n_components=m, random_state=0)
    return model.fit_transform(X)

def fa(X, m):
    model = cluster.FeatureAgglomeration(n_clusters=m)
    model.fit(X)
    return model.transform(X)

if __name__ == '__main__':

    X_train, y_train, X_test, y_test = preprocessing.preprocess_NN()

    # Control
    if False:
        path = 'params/control.obj'
        if False:
            parameters = get_best_parameters(X_train, y_train, path)
        filehandler = open(path, 'rb')
        parameters = pickle.load(filehandler)
        print(parameters)
        evaluate(X_train, y_train, X_test, y_test, parameters)

    # PCA
    if False:
        path = 'params/PCA.obj'
        X_train_pca = pca(X_train)
        if False:
            parameters = get_best_parameters(X_train_pca, y_train, path)
        filehandler = open(path, 'rb')
        parameters = pickle.load(filehandler)
        print(parameters)
        X_test_pca = pca(X_test)
        evaluate(X_train_pca, y_train, X_test_pca, y_test, parameters)

    # ICA
    if False:
        path = 'params/ICA.obj'
        X_train_ica = ica(X_train, 12)
        if False:
            parameters = get_best_parameters(X_train_ica, y_train, path)
        filehandler = open(path, 'rb')
        parameters = pickle.load(filehandler)
        print(parameters)
        X_test_ica = ica(X_test, 12)
        evaluate(X_train_ica, y_train, X_test_ica, y_test, parameters)

    # RP
    if False:
        path = 'params/RP.obj'
        X_train_rp = rp(X_train, 14)
        if False:
            parameters = get_best_parameters(X_train_rp, y_train, path)
        filehandler = open(path, 'rb')
        parameters = pickle.load(filehandler)
        print(parameters)
        X_test_rp = rp(X_test, 14)
        evaluate(X_train_rp, y_train, X_test_rp, y_test, parameters)

    # FA
    if True:
        path = 'params/FA.obj'
        X_train_fa = rp(X_train, 5)
        if False:
            parameters = get_best_parameters(X_train_fa, y_train, path)
        filehandler = open(path, 'rb')
        parameters = pickle.load(filehandler)
        print(parameters)
        X_test_fa = fa(X_test, 5)
        evaluate(X_train_fa, y_train, X_test_fa, y_test, parameters)

    # K-means clustering

    # EM