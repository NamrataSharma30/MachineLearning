import numpy as np
import math
import time
from sklearn.tree import DecisionTreeRegressor
from gradient_descent_logistic_regression import utils as ut
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.close('all')

    data_path = '../datasets/'  # Use your own path here

    X = np.load(data_path + 'particles_X.npy').astype(np.float32)
    y = np.load(data_path + 'particles_y.npy').astype(np.float32)

    X_train, X_test, y_train, y_test = ut.split_train_test(X, y, seed=20)

    depths = []
    times = []
    mse_train = []
    mse_test = []
    for max_depth in range(1, 25):
        depths.append(max_depth)
        print('\nMaximum depth:', max_depth)
        model = DecisionTreeRegressor(splitter="random", max_depth=max_depth, min_samples_split=10, min_impurity_decrease=5.0)
        start = time.time()
        model.fit(X_train, y_train)
        elapsed_time = time.time() - start
        print('Elapsed_time training  {0:.4f} secs'.format(elapsed_time))
        times.append(elapsed_time)

        print('Performance on training set')
        start = time.time()
        pred = model.predict(X_train)
        elapsed_time = time.time() - start
        print('Elapsed_time testing  {0:.4f} secs'.format(elapsed_time))
        err = ut.mse(y_train, pred)
        mse_train.append(err)
        print('Mean-squared error: {0:.6f}'.format(err))

        print('Performance on test set')
        start = time.time()
        pred = model.predict(X_test)
        elapsed_time = time.time() - start
        print('Elapsed_time testing  {0:.4f} secs'.format(elapsed_time))
        err = ut.mse(y_test, pred)
        mse_test.append(err)
        print('Mean-squared error: {0:.6f}'.format(err))

    fig, ax = plt.subplots(2, figsize=(6, 6))

    ax[0].plot(depths, times)
    ax[0].title.set_text('Training time')
    ax[1].plot(depths, mse_train)
    ax[1].plot(depths, mse_test)
    ax[1].title.set_text('Mean squared error')
