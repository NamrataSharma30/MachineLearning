import numpy as np
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import random
import cmath
from gradient_descent_logistic_regression import utils as ut

def mse(p, y):
    return np.mean((p - y) ** 2)


if __name__ == "__main__":
    plt.close('all')

    X_train = np.load('x_train.npy')
    X_test = np.load('x_test.npy')
    y_train = np.load('y_train.npy')


    def synthetic_regression_x(x1, x2):
        w = random.randrange(0, 1)
        return x1 * w + x2 * (1 - w)


    def synthetic_regression_y(y1, y2):
        w = random.randrange(0, 1)
        return y1 * w + y2 * (1 - w)


    print(X_train.shape, y_train.shape)

    temp = []
    for i in range(1000):
        for j in range(80000):
            if j not in temp:
                if cmath.isclose(y_train[i], y_train[j], rel_tol=1e-05, abs_tol=0.0):
                    temp.append(j)
                    X_train = np.vstack((X_train, synthetic_regression_x(X_train[i], X_train[j])))
                    y_train = np.vstack((y_train, synthetic_regression_y(y_train[i], y_train[j])))
        print(i)
    print(X_train.shape, y_train.shape)

    X_train, X_test, y_train, y_test = ut.split_train_test(X_train, y_train)

    depths = []
    times = []
    mse_train = []
    mse_test = []
    for max_depth in range(1, 30):
        depths.append(max_depth)
        print('\nMaximum depth:', max_depth)
        model = RandomForestRegressor(max_depth=max_depth, random_state=0)
        start = time.time()
        model.fit(X_train, y_train.reshape(-1))
        elapsed_time = time.time() - start
        print('Elapsed_time training  {0:.4f} secs'.format(elapsed_time))
        times.append(elapsed_time)

        print('Performance on test set')
        start = time.time()
        pred = model.predict(X_test)
        elapsed_time = time.time() - start
        print('Elapsed_time testing  {0:.4f} secs'.format(elapsed_time))

    ids = np.arange(start=1., stop=X_test.shape[0] + 1, dtype=float)
    submission_df_1 = pd.DataFrame({'ID': ids, 'Prediction': pred})
    submission_df_1.to_csv('randomregressor_submission_2.csv', index=False)
    print('Mean-squared error: {0:.6f}'.format(mse(pred, y_test)))
