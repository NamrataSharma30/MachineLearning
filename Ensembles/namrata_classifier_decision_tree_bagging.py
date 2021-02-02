import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from utils import *
import math
import time
from gradient_descent_logistic_regression import utils as ut
import matplotlib.pyplot as plt

if __name__ == "__main__":

    print('ENSEMBLE WITH RANDOM TRAINING EXAMPLE SELECTION')
    plt.close('all')

    data_path = '../datasets/'  # Use your own path here

    # X = np.load(data_path + 'particles_X.npy')[:10000]
    # y = np.load(data_path + 'particles_y.npy')[:10000]
    X = np.load(data_path + 'mnist_X.npy').reshape(-1, 28 * 28)
    y = np.load(data_path + 'mnist_y.npy')
    X_train, X_test, y_train, y_test = ut.split_train_test(X, y, seed=20)

    ensemble = []
    start = time.time()
    ensemble_size = 11
    for i in range(ensemble_size):
        print('Training ensemble model', i)
        model = RandomForestClassifier(n_estimators=200, criterion="entropy", random_state=0)
        a = np.random.randint(low=0, high=X_train.shape[0], size=X_train.shape[0])
        model.fit(X_train[a], y_train[a])
        ensemble.append(model)
        elapsed_time = time.time() - start
        print('Elapsed time training so far {:.6f} secs'.format(elapsed_time))

    start = time.time()
    votes = np.zeros((X_test.shape[0], np.amax(y) + 1), dtype=int)
    row_ind = np.arange(y_test.shape[0])
    for i, model in enumerate(ensemble):
        pred = model.predict(X_test)
        print('Model {} accuracy: {:.6f}'.format(i, ut.accuracy(y_test, pred)))
        votes[row_ind, pred] += 1

    ens_pred = np.argmax(votes, axis=1)
    elapsed_time = time.time() - start
    print('Ensemble accuracy: {:.6f}'.format(ut.accuracy(y_test, ens_pred)))
    print('Elapsed time testing {:.6f} secs'.format(elapsed_time))
