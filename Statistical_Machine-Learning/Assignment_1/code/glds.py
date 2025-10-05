import numpy as np
from typing import Tuple
from scipy.stats import multivariate_normal

def predict_step(
    mu_t_minus_1: np.ndarray,
    Sigma_t_minus_1: np.ndarray,
    A: np.ndarray,
    Q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform the predict step of the Kalman filter.

    Parameters:
    -----------
    mu_t_minus_1: np.ndarray, shape (Ds,)
        The mean of the state at time t-1 given observations up to time t-1
    Sigma_t_minus_1: np.ndarray, shape (Ds, Ds)
        The covariance of the state at time t-1 given observations up to time t-1
    A: np.ndarray, shape (Ds, Ds)
        The state transition matrix
    Q: np.ndarray, shape (Ds, Ds)
        The covariance of the dynamics noise

    Returns:
    --------
    mu_t_given_t_minus_1: np.ndarray, shape (Ds,)
        The mean of the state at time t given observations up to time t-1
    Sigma_t_given_t_minus_1: np.ndarray, shape (Ds, Ds)
        The covariance of the state at time t given observations up to time t-1
    """
    # Question 1.3 Implement the predict step, p(s_t|y_{1:t-1})
    # Hint: p(s_t|y_{1:t-1}) = N(s_t; mu_t|t-1, Sigma_t|t-1)

    # Initialize output variables
    mu_t_given_t_minus_1 = A @ mu_t_minus_1
    Sigma_t_given_t_minus_1 = A @ Sigma_t_minus_1 @ A.T + Q

    return mu_t_given_t_minus_1, Sigma_t_given_t_minus_1


def update_step(
    mu_t_given_t_minus_1: np.ndarray,
    Sigma_t_given_t_minus_1: np.ndarray,
    y_t: np.ndarray,
    C: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform the update step of the Kalman filter.

    Parameters:
    -----------
    mu_t_given_t_minus_1: np.ndarray, shape (Ds,)
        The mean of the state at time t given observations up to time t-1
    Sigma_t_given_t_minus_1: np.ndarray, shape (Ds, Ds)
        The covariance of the state at time t given observations up to time t-1
    y_t: np.ndarray, shape (Dy,)
        The observation at time t
    C: np.ndarray, shape (Dy, Ds)
        The observation matrix
    R: np.ndarray, shape (Dy, Dy)
        The covariance of the observation noise

    Returns:
    --------
    mu_t_given_t: np.ndarray, shape (Ds,)
        The mean of the state at time t given observations up to time t
    Sigma_t_given_t: np.ndarray, shape (Ds, Ds)
        The covariance of the state at time t given observations up to time t
    """
    # Question 1.3 Implement the update step
    # Hint: p(s_t|y_{1:t}) = N(s_t; mu_t|t, Sigma_t|t)

    S = C @ Sigma_t_given_t_minus_1 @ C.T + R
    K = Sigma_t_given_t_minus_1 @ C.T @ np.linalg.inv(S)

    innov = y_t - C @ mu_t_given_t_minus_1

    # Initialize output variables
    mu_t_given_t = mu_t_given_t_minus_1 + K @ innov
    Sigma_t_given_t = Sigma_t_given_t_minus_1 - K @ C @ Sigma_t_given_t_minus_1


    return mu_t_given_t, Sigma_t_given_t


def predictive_distribution(
    mu_t_minus_1: np.ndarray,
    Sigma_t_minus_1: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean and covariance of the predictive distribution p(y_t|y_{1:t-1}).

    Parameters:
    -----------
    mu_t_minus_1: np.ndarray, shape (Ds,)
        The mean of the state at time t-1 given observations up to time t-1
    Sigma_t_minus_1: np.ndarray, shape (Ds, Ds)
        The covariance of the state at time t-1 given observations up to time t-1
    A: np.ndarray, shape (Ds, Ds)
        The state transition matrix
    C: np.ndarray, shape (Dy, Ds)
        The observation matrix
    Q: np.ndarray, shape (Ds, Ds)
        The covariance of the dynamics noise
    R: np.ndarray, shape (Dy, Dy)
        The covariance of the observation noise

    Returns:
    --------
    pred_mean: np.ndarray, shape (Dy,)
        The mean of the predictive distribution p(y_t|y_{1:t-1})
    pred_cov: np.ndarray, shape (Dy, Dy)
        The covariance of the predictive distribution p(y_t|y_{1:t-1})
    """
    # Question 1.4 Implement the predictive distribution
    # Hint: p(y_t|y_{1:t-1}) = N(y_t; pred_mean, pred_cov)

    mu_t_prior = A @ mu_t_minus_1
    sigma_t_prior = A @ Sigma_t_minus_1 @ A.T + Q

    # Initialize output variables
    pred_mean = C @ mu_t_prior
    pred_cov = C @ sigma_t_prior @ C.T + R

    return pred_mean, pred_cov


def compute_log_marginal_likelihood(
    y: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    mu_0: np.ndarray,
    Sigma_0: np.ndarray,
) -> float:
    """
    Calculate the log marginal likelihood log p(y_{1:T}).

    Parameters:
    -----------
    y: np.ndarray, shape (T, Dy)
        The observations from time 1 to time T
    A: np.ndarray, shape (Ds, Ds)
        The state transition matrix
    C: np.ndarray, shape (Dy, Ds)
        The observation matrix
    Q: np.ndarray, shape (Ds, Ds)
        The covariance of the dynamics noise
    R: np.ndarray, shape (Dy, Dy)
        The covariance of the observation noise
    mu_0: np.ndarray, shape (Ds,)
        The mean of the initial state
    Sigma_0: np.ndarray, shape (Ds, Ds)
        The covariance of the initial state

    Returns:
    --------
    log_likelihood: float
        The log marginal likelihood log p(y_{1:T})
    """
    # Question 1.5 Implement the log marginal likelihood computation

    # Initialize variables
    T = y.shape[0]
    log_likelihood = 0.0

    # Current state estimate
    mu_t_minus_1 = mu_0
    Sigma_t_minus_1 = Sigma_0

    for t in range(T):
        # predict step
        mu_t_prior = A @ mu_t_minus_1
        sigma_t_prior = A @ Sigma_t_minus_1 @ A.T + Q

        # observation prediction
        pred_mean = C @ mu_t_prior
        pred_cov = C @ sigma_t_prior @ C.T + R

        # log likelihood
        log_likelihood += multivariate_normal.logpdf(y[t], mean=pred_mean, cov=pred_cov)

        # update step
        K = sigma_t_prior @ C.T @ np.linalg(pred_cov)
        mu_t = mu_t_prior + K @ (y[t] - pred_mean)
        sigma_t = sigma_t_prior - K @ C @ sigma_t_prior

        mu_t_minus_1 = mu_t
        Sigma_t_minus_1 = sigma_t

    return log_likelihood


# DO NOT MODIFY THE CODE BELOW!
# This code is provided to you for running the Kalman filter
# on a sequence of observations.
def kalman_filter(
    y: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    mu_0: np.ndarray,
    Sigma_0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the Kalman filter on a sequence of observations.

    Parameters:
    -----------
    y: np.ndarray, shape (T, Dy)
        The observations from time 1 to time T
    A: np.ndarray, shape (Ds, Ds)
        The state transition matrix
    C: np.ndarray, shape (Dy, Ds)
        The observation matrix
    Q: np.ndarray, shape (Ds, Ds)
        The covariance of the dynamics noise
    R: np.ndarray, shape (Dy, Dy)
        The covariance of the observation noise
    mu_0: np.ndarray, shape (Ds,)
        The mean of the initial state
    Sigma_0: np.ndarray, shape (Ds, Ds)
        The covariance of the initial state

    Returns:
    --------
    filtered_means: np.ndarray, shape (T, Ds)
        The means of the filtered states p(s_t|y_{1:t}) for t=1...T
    filtered_covs: np.ndarray, shape (T, Ds, Ds)
        The covariances of the filtered states p(s_t|y_{1:t}) for t=1...T
    """
    # Initialize variables
    T = y.shape[0]
    Ds = A.shape[0]
    filtered_means = np.zeros((T, Ds))
    filtered_covs = np.zeros((T, Ds, Ds))

    # Start with initial state
    mu_prev = mu_0
    Sigma_prev = Sigma_0

    # Run Kalman filter
    for t in range(T):
        # Predict step
        mu_pred, Sigma_pred = predict_step(mu_prev, Sigma_prev, A, Q)

        # Update step
        mu_filt, Sigma_filt = update_step(mu_pred, Sigma_pred, y[t], C, R)

        # Store results
        filtered_means[t] = mu_filt
        filtered_covs[t] = Sigma_filt

        # Update for next iteration
        mu_prev = mu_filt
        Sigma_prev = Sigma_filt

    return filtered_means, filtered_covs
