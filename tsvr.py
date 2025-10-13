import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel


class TwinSVR:
    def __init__(self, C1=1.0, C2=1.0, kernel='rbf', gamma=1.0):
        self.C1 = C1
        self.C2 = C2
        self.kernel = kernel
        self.gamma = gamma

    def _kernel(self, X1, X2):
        if self.kernel == 'rbf':
            return rbf_kernel(X1, X2, gamma=self.gamma)
        else:
            raise ValueError("Unsupported kernel")

    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y
        K = self._kernel(X, X)
        I = np.eye(len(K))

        self.alpha1_ = np.linalg.solve(K.T @ K + (1/self.C1) * I, K.T @ y)
        self.alpha2_ = np.linalg.solve(K.T @ K + (1/self.C2) * I, K.T @ y)
        self.b1_ = np.mean(y - K @ self.alpha1_)
        self.b2_ = np.mean(y - K @ self.alpha2_)
        return self

    def predict(self, X):
        K = self._kernel(X, self.X_train_)
        f1 = K @ self.alpha1_ + self.b1_
        f2 = K @ self.alpha2_ + self.b2_
        y_pred = 0.5 * (f1 + f2)
        return y_pred.reshape(-1, 1)   # selalu 2D