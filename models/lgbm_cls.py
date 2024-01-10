from scipy.sparse import load_npz
import numpy as np

X_train = load_npz("../data/X_train.npz")
print("The shape of X_train is:", X_train.shape)
y_train = np.load("../data/y_train.npy")
print("The shape of y_train is:", y_train.shape)
X_test = load_npz("../data/X_test.npz")
print("The shape of X_test is:", X_test.shape)
print(X_test.toarray())
