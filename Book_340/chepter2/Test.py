import mglearn
import matplotlib
matplotlib.matplotlib_fname()
import matplotlib.pyplot as plt

import numpy as np

X, y = mglearn.datasets.make_forge()
plt.scatter(X[:, 0], X[:, 1], c=y, s=60, cmap=mglearn.cm2)
plt.show()
print("X.shape: %s" % (X.shape,))

X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.plot(X, -3 * np.ones(len(X)), 'o')
plt.ylim(-3.1, 3.1)
plt.show()
