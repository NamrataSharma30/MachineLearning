import numpy as np
from gradient_descent_logistic_regression import utils  as ut
import math
import time
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.close('all')

    data_path = '../datasets/'  # Use your own path here

    X = np.load(data_path + 'mnist_X.npy').reshape(-1, 28 * 28)

    y = np.load(data_path + 'mnist_y.npy')
    thr = 127
    X = (X > thr).astype(int)

    X_train, X_test, y_train, y_test = ut.split_train_test(X, y, seed=20)

    model = DecisionTreeClassifier(random_state=0, criterion="entropy", splitter="random")

    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start
    print('Elapsed_time training  {0:.6f} secs'.format(elapsed_time))

    pred = model.predict(X_train)
    print('Accuracy on training set: {0:.6f}'.format(ut.accuracy(y_train, pred)))

    start = time.time()
    pred = model.predict(X_test)
    elapsed_time = time.time() - start
    print('Elapsed_time testing  {0:.6f} secs'.format(elapsed_time))
    print('Accuracy on test set: {0:.6f}'.format(ut.accuracy(y_test, pred)))
