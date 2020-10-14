import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from solutionAreaGraph import plot_decision_regions

# First steps with scikit-learn
# Loading the Iris dataset from scikit-learn.

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print('Метки классов:', np.unique(y))

# Splitting data into 70% training and 30% test data:

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

print('Количество меток в y:', np.bincount(y))
print('Количество меток в y_train:', np.bincount(y_train))
print('Количество меток в y_test:', np.bincount(y_test))

# Standardizing the features:

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Training a perceptron via scikit-learn base perceptron

ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print(f'Неправильно классифицированных образцов: {(y_test != y_pred).sum()} % ')

print(f'Правильность: {accuracy_score(y_test, y_pred)} %')
print(f'Правильность: {ppn.score(X_test_std, y_test)} %')

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))
plt.xlabel('длина лепестка [стандартизированная]')
plt.ylabel('ширина лепестка [стандартизированная]')
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('images/sklearnTrain/solution_area_graph.png', dpi=300)
plt.show()


# sigmoid function

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)

plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')

# y axis ticks and gridline
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)

plt.tight_layout()
plt.savefig('images/sklearnTrain/sigmoid_function.png', dpi=300)
plt.show()
