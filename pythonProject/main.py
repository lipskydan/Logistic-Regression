import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

# First steps with scikit-learn
# Loading the Iris dataset from scikit-learn.

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print('Class labels:', np.unique(y))


# Splitting data into 70% training and 30% test data:

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

# class LogisticRegressionGD(object):
#     """Logistic Regression Classifier using gradient descent.
#
#     Parameters
#     ------------
#     eta : float
#       Learning rate (between 0.0 and 1.0)
#     n_iter : int
#       Passes over the training dataset.
#     random_state : int
#       Random number generator seed for random weight
#       initialization.
#
#
#     Attributes
#     -----------
#     w_ : 1d-array
#       Weights after fitting.q
#     cost_ : list
#       Sum-of-squares cost function value in each epoch.
#
#     """
#
#     def __init__(self, eta=0.05, n_iter=100, random_state=1):
#         self.eta = eta
#         self.n_iter = n_iter
#         self.random_state = random_state
#
#     def fit(self, X, y):
#         """ Fit training data.
#
#         Parameters
#         ----------
#         X : {array-like}, shape = [n_samples, n_features]
#           Training vectors, where n_samples is the number of samples and
#           n_features is the number of features.
#         y : array-like, shape = [n_samples]
#           Target values.
#
#         Returns
#         -------
#         self : object
#
#         """
#         rgen = np.random.RandomState(self.random_state)
#         self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
#         self.cost_ = []
#
#         for _ in range(self.n_iter):
#             net_input = self.net_input(X)
#             output = self.activation(net_input)
#             errors = (y - output)
#             self.w_[1:] += self.eta * X.T.dot(errors)
#             self.w_[0] += self.eta * errors.sum()
#             cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
#             self.cost_.append(cost)
#         return self
#
#     def net_input(self, X):
#         """Calculate net input"""
#         return np.dot(X, self.w_[1:]) + self.w_[0]
#
#     def activation(self, z):
#         """Compute logistic sigmoid activation"""
#         return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
#
#     def predict(self, X):
#         """Return class label after unit step"""
#         # return np.where(self.net_input(X) >= 0.0, 1, 0)
#         # equivalent to:
#         return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
#
#
