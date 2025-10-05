import numpy as np
import matplotlib.pyplot as plt

np.random.seed(4600)

from glds import kalman_filter

A = np.array(
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
C = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
Q = 0.1 * np.eye(4)
R = 0.5 * np.eye(2)
mu_0 = np.array([10, 10, 1, -1])
Sigma_0 = 0.5 * np.eye(4)


def sample_glds(T, A, C, Q, R, mu_0, Sigma_0):
    x = np.zeros((T, 4))
    y = np.zeros((T, 2))
    x[0] = np.random.multivariate_normal(mu_0, Sigma_0)
    y[0] = np.dot(C, x[0]) + np.random.multivariate_normal(np.zeros(2), R)
    for t in range(1, T):
        x[t] = np.random.multivariate_normal(np.dot(A, x[t - 1]), Q)
        y[t] = np.random.multivariate_normal(np.dot(C, x[t]), R)
    return x, y


T = 20
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
for i in range(2):
    s, y = sample_glds(T, A, C, Q, R, mu_0, Sigma_0)
    ax = axs[i]
    ax.plot(y[:, 0], y[:, 1], "ro-", label="observed data")
    ax.plot(s[:, 0], s[:, 1], "bx--", label="hidden state")
    ax.plot(s[0, 0], s[0, 1], "g*", ms=12, label="initial state")
    ax.plot(s[-1, 0], s[-1, 1], "r*", ms=12, label="final state")

    # uncomment the following to see the estimated state
    s_filter, cov_filter = kalman_filter(y, A, C, Q, R, mu_0, Sigma_0)
    s_filter_var = np.sqrt(np.diagonal(cov_filter, axis1=1, axis2=2))
    ax.errorbar(
        s_filter[:, 0],
        s_filter[:, 1],
        xerr=s_filter_var[:, 0],
        yerr=s_filter_var[:, 1],
        fmt="kx--",
        label="estimated state",
        alpha=0.5
    )

    ax.set_xlabel(r"$y_1$")
    ax.set_ylabel(r"$y_2$")
    ax.legend()

fig.savefig("/tmp/glds_example.pdf", bbox_inches="tight", dpi=300, pad_inches=0)
plt.show()