from jax import random
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.stats import norm, multivariate_normal


# Generate synthetic 2D data
def generate_data(data_key, n_samples=200, n_features=2):
    # Generate features
    X = random.normal(data_key, (n_samples, n_features))

    # True weights and bias
    true_weights = jnp.array([1.5, -2.0])
    true_bias = 0.5

    # Compute logits
    logits = jnp.dot(X, true_weights) + true_bias

    # Convert logits to probabilities
    probs = 1.0 / (1.0 + jnp.exp(-logits))

    # Generate binary labels
    label_key = random.split(data_key)[0]
    y = random.bernoulli(label_key, probs).astype(jnp.float32)

    return X, y, true_weights, true_bias


def plot_predictions(hmc_samples, vi_samples, X, y):
    # Function to create prediction grid
    def create_prediction_grid(x_min, x_max, y_min, y_max, samples, grid_size=100):
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        xx, yy = np.meshgrid(x_grid, y_grid)
        X_grid = np.column_stack([xx.flatten(), yy.flatten()])

        # Compute probabilities for each sample
        W = samples["weights"]
        b = samples["bias"]

        all_probs = []

        for i in range(W.shape[0]):
            logits = X_grid @ W[i] + b[i]
            probs = 1.0 / (1.0 + np.exp(-logits))
            all_probs.append(probs)

        all_probs = np.stack(all_probs)
        avg_probs = np.mean(all_probs, axis=0)

        return xx, yy, avg_probs.reshape(xx.shape)

    # Prepare data for plotting
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create prediction grids
    xx_hmc, yy_hmc, prob_hmc = create_prediction_grid(
        x_min, x_max, y_min, y_max, hmc_samples
    )
    xx_vi, yy_vi, prob_vi = create_prediction_grid(
        x_min, x_max, y_min, y_max, vi_samples
    )

    # Plot predictions and differences between HMC and VI
    plt.figure(figsize=(15, 5))

    # HMC Predictions
    plt.subplot(1, 3, 1)
    cmap = ListedColormap(["#FFAAAA", "#AAAAFF"])
    plt.contourf(xx_hmc, yy_hmc, prob_hmc > 0.5, cmap=cmap, alpha=0.5)
    plt.contour(
        xx_hmc,
        yy_hmc,
        prob_hmc,
        levels=[0.25, 0.5, 0.75],
        colors="k",
        linestyles="--",
    )
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors="k", s=40)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("HMC Predictions")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # VI Predictions
    plt.subplot(1, 3, 2)
    plt.contourf(xx_vi, yy_vi, prob_vi > 0.5, cmap=cmap, alpha=0.5)
    plt.contour(
        xx_vi,
        yy_vi,
        prob_vi,
        levels=[0.25, 0.5, 0.75],
        colors="k",
        linestyles="--",
    )
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors="k", s=40)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("VI Predictions")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # Difference between HMC and VI predictions
    plt.subplot(1, 3, 3)
    diff = np.abs(prob_hmc - prob_vi)
    plt.contourf(xx_hmc, yy_hmc, diff, cmap="viridis")
    plt.colorbar(label="Absolute Difference")
    plt.contour(
        xx_hmc,
        yy_hmc,
        diff,
        levels=[0.05, 0.1, 0.2],
        colors="w",
        linestyles="--",
    )
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors="k", s=40)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("Difference: |HMC - VI|")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.savefig(
        "/tmp/hmc_vs_vi_predictions.pdf",
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.0,
    )


def plot_posterior(hmc_samples, vi_posterior, true_weights, true_bias):
    # Labels for the parameters
    param_names = ["Weight 1", "Weight 2", "Bias"]
    samples = [
        hmc_samples["weights"][:, 0],
        hmc_samples["weights"][:, 1],
        hmc_samples["bias"],
    ]
    true_values = [true_weights[0], true_weights[1], true_bias]
    vi_means = vi_posterior.mean
    vi_variances = vi_posterior.variance

    fig, axes = plt.subplots(3, 3, figsize=(20, 20))

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            if i == j:
                # Plot HMC histogram
                sns.histplot(
                    samples[i],
                    ax=ax,
                    kde=True,
                    color="royalblue",
                    label="HMC",
                    stat="density",
                )

                # Plot VI density curve
                x_range = np.linspace(
                    samples[i].min() - 0.5, samples[i].max() + 0.5, 1000
                )
                vi_mean = vi_means[i]
                vi_std = jnp.sqrt(vi_variances[i])
                vi_pdf = norm.pdf(x_range, vi_mean, vi_std)
                ax.plot(x_range, vi_pdf, "r-", lw=2, label="VI")

                # Add true value
                ax.axvline(true_values[i], color="k", linestyle="--", label="True")
                # ax.set_title(f"{param_names[i]} Marginal", fontsize=8)
            else:
                # Plot HMC samples
                sns.kdeplot(
                    x=samples[j],
                    y=samples[i],
                    ax=ax,
                    cmap="Blues",
                    fill=True,
                    thresh=0.05,
                )
                ax.scatter(
                    samples[j],
                    samples[i],
                    alpha=0.3,
                    s=5,
                    color="royalblue",
                    label="HMC",
                )
                # plot 2D VI contours
                x_range = np.linspace(
                    samples[j].min() - 0.5, samples[j].max() + 0.5, 100
                )
                y_range = np.linspace(
                    samples[i].min() - 0.5, samples[i].max() + 0.5, 100
                )
                X, Y = np.meshgrid(x_range, y_range)
                pos = np.dstack((X, Y))
                rv = multivariate_normal(
                    [vi_means[j], vi_means[i]],
                    [[vi_variances[j], 0], [0, vi_variances[i]]],
                )
                Z = rv.pdf(pos)
                ax.contour(X, Y, Z, levels=5, colors="red", alpha=0.7)
                ax.scatter(
                    true_values[j],
                    true_values[i],
                    color="k",
                    marker="x",
                    s=100,
                    label="True",
                )

                # if i == 2:
                #     ax.legend()

            if i == 2:
                ax.set_xlabel(param_names[j])
            if j == 0:
                ax.set_ylabel(param_names[i])

    fig.savefig(
        "/tmp/hmc_vs_vi_posterior.pdf", bbox_inches="tight", dpi=300, pad_inches=0.0
    )
