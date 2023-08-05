from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
class Dataset:
    def __init__(self):
        X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
        X = X.astype(np.float32)
        X /= 255.
        X -= X.mean(axis=0)
        encoder = OneHotEncoder(sparse=False)
        y = encoder.fit_transform(y.values.reshape(-1, 1))
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            X.astype(np.float32),
            y.astype(np.float32),
            test_size=(1 / 7.))
    def data(self):
        self.train_x = self.train_x.to_numpy()
        self.test_x = self.test_x.to_numpy()
        return self.train_x, self.test_x, self.train_y, self.test_y
if __name__ == '__main__':
    train_x, test_x, train_y, test_y = Dataset().data()