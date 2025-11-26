import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ======================== بخش 1 ========================

np.random.seed(42)
A = np.array([
    [1, 2.0],
    [1, 3.0],
    [1, 4.5],
    [1, 5.5]
])

y = np.array([
    [3.1],
    [4.9],
    [8.2],
    [10.1]
])
m = len(y)

AtA = A.T @ A
Aty = A.T @ y
theta_normal = np.linalg.inv(AtA) @ Aty
print("1.1 theta_normal =", theta_normal.ravel())

learning_rate = 0.01
n_iterations = 1000
theta_bgd = np.zeros((A.shape[1], 1))

for _ in range(n_iterations):
    grad = (1/m) * (A.T @ (A @ theta_bgd - y))
    theta_bgd -= learning_rate * grad

print("1.2 theta_bgd    =", theta_bgd.ravel())

U, S, Vt = np.linalg.svd(A, full_matrices=False)
S_inv = np.diag(1.0 / S)
A_pinv = Vt.T @ S_inv @ U.T
theta_svd = A_pinv @ y
print("1.3 theta_svd    =", theta_svd.ravel())

noise = np.random.rand(m, 1) * 0.0001
A_collinear = np.hstack((A, A[:, [1]] + noise))

AtA_c = A_collinear.T @ A_collinear
Aty_c = A_collinear.T @ y
theta_normal_c = np.linalg.inv(AtA_c) @ Aty_c
print("1.4 theta_normal_collinear =", theta_normal_c.ravel())

U_c, S_c, Vt_c = np.linalg.svd(A_collinear, full_matrices=False)
S_inv_c = np.diag(1.0 / S_c)
A_pinv_c = Vt_c.T @ S_inv_c @ U_c.T
theta_svd_c = A_pinv_c @ y
print("1.4 theta_svd_collinear    =", theta_svd_c.ravel())

# =================== بخش 2 ===================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ["MPG", "Cylinders", "Displacement", "Horsepower",
                "Weight", "Acceleration", "ModelYear", "Origin", "CarName"]
data_mpg = pd.read_csv(url, names=column_names,
                       na_values="?", delim_whitespace=True)
data_mpg = data_mpg.dropna(subset=["Horsepower"])
data_mpg["Horsepower"] = data_mpg["Horsepower"].astype(float)

y = data_mpg["MPG"].values.reshape(-1, 1)
x = data_mpg["Horsepower"].values.reshape(-1, 1)
m = len(y)

A = np.hstack([np.ones((m, 1)), x]).astype(float)
A_norm = A.copy()
A_norm[:, 1] = (A_norm[:, 1] - A_norm[:, 1].mean()) / A_norm[:, 1].std()

def cost(A, y, theta):
    e = A @ theta - y
    return (e**2).mean() / 2.0

np.random.seed(0)
theta_sgd = np.zeros((2, 1))
alpha_sgd = 0.01
n_iter_sgd = 5000
J_sgd = []

for _ in range(n_iter_sgd):
    i = np.random.randint(m)
    Ai = A_norm[i:i+1]
    yi = y[i:i+1]
    grad_i = (Ai @ theta_sgd - yi) * Ai.T
    theta_sgd -= alpha_sgd * grad_i
    J_sgd.append(cost(A_norm, y, theta_sgd))

theta_bgd2 = np.zeros((2, 1))
alpha_bgd = 0.01
n_iter_bgd = 1000
J_bgd = []

for _ in range(n_iter_bgd):
    grad = (A_norm.T @ (A_norm @ theta_bgd2 - y)) / m
    theta_bgd2 -= alpha_bgd * grad
    J_bgd.append(cost(A_norm, y, theta_bgd2))

print("2.3 theta_sgd =", theta_sgd.ravel())
print("2.3 theta_bgd =", theta_bgd2.ravel())

plt.figure()
plt.plot(range(len(J_bgd)), J_bgd, label="BGD")
plt.plot(range(len(J_sgd)), J_sgd, label="SGD")
plt.xlabel("Iteration")
plt.ylabel("J(theta)")
plt.legend()
plt.title("BGD vs SGD (Auto MPG)")
plt.show()

# ======================= بخش 3 =======================

y = data_mpg["MPG"].values.reshape(-1, 1)
x = data_mpg["Horsepower"].values.reshape(-1, 1)
m = len(y)

A_poly = np.hstack([np.ones((m, 1)), x, x**2]).astype(float)
mu = A_poly[:, 1:].mean(axis=0)
sigma = A_poly[:, 1:].std(axis=0)
A_poly[:, 1:] = (A_poly[:, 1:] - mu) / sigma

np.random.seed(0)
theta_poly = np.zeros((3, 1))
alpha_poly = 0.01
n_iter_poly = 5000

for _ in range(n_iter_poly):
    i = np.random.randint(m)
    Ai = A_poly[i:i+1]
    yi = y[i:i+1]
    error = Ai @ theta_poly - yi
    grad = Ai.T * error
    theta_poly -= alpha_poly * grad

print("3.3 theta_poly =", theta_poly.ravel())

xp = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
Ap = np.hstack([np.ones((len(xp), 1)), xp, xp**2]).astype(float)
Ap[:, 1:] = (Ap[:, 1:] - mu) / sigma
yp = Ap @ theta_poly

plt.figure()
plt.scatter(x, y, s=10, label="data")
plt.plot(xp, yp, label="poly fit")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.legend()
plt.title("Nonlinear Regression (Quadratic)")
plt.show()
