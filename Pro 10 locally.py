import numpy as np
import matplotlib.pyplot as plt

X = np.array([1,2,3,4,5,6,7,8,9,10])
Y = np.array([2,4,5,4,5,7,8,7,9,10])

def lwr(x):
    w = np.exp(-(X-x)**2/2)
    A = np.c_[np.ones(len(X)), X]
    W = np.diag(w)
    theta = np.linalg.pinv(A.T @ W @ A) @ A.T @ W @ Y
    return np.array([1,x]) @ theta

x_test = np.linspace(1,10,100)
y_pred = [lwr(x) for x in x_test]

plt.scatter(X,Y)
plt.plot(x_test,y_pred)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Locally Weighted Regression")
plt.show()
