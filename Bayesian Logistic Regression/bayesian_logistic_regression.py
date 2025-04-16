import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide
import matplotlib.pyplot as plt

from utils import generate_data, plot_predictions, plot_posterior

# Set a random seed for reproducibility
SEED = 42
rng_key = random.PRNGKey(SEED)
rng_key, data_key = random.split(rng_key)

# Generate the dataset
X, y, true_weights, true_bias = generate_data(data_key)


# Define the logistic regression model
def logistic_regression(X, y=None):
    # Define priors for weights and bias
    n_features = X.shape[1]

    ### P(\theta) = P(weights, bias) = P(weights) * P(bias)

    # Prior for weights: Normal(0, 1)
    weights = numpyro.sample("weights", dist.Normal(0.0, 1.0).expand([n_features]))

    # Prior for bias: Normal(0, 1)
    bias = numpyro.sample("bias", dist.Normal(0.0, 1.0))

    # Likelihood (observed data)
    # p(y | \theta, X) = Bernoulli(logits = X * weights + bias)
    # Linear combination
    logits = jnp.dot(X, weights) + bias

    # Likelihood (observed data)
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

    # Store predictions for later use
    probs = jax.nn.sigmoid(logits)
    numpyro.deterministic("probs", probs)


# Run HMC (NUTS)
def run_hmc(model, X, y, rng_key):
    # Initialize NUTS kernel
    nuts_kernel = NUTS(model)

    # Initialize MCMC
    mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000)

    # Run MCMC
    mcmc.run(rng_key, X, y)

    return mcmc


# Run variational inference
def run_vi(model, X, y, rng_key):
    # Define the variational distribution
    var_dist = autoguide.AutoDiagonalNormal(model)

    # Initialize stochastic variational inference
    optimizer = numpyro.optim.Adam(step_size=0.01)
    svi = SVI(model, var_dist, optimizer, loss=Trace_ELBO())

    # optimise
    svi_result = svi.run(rng_key, 5000, X, y)

    return svi_result, var_dist


# Run both inference methods
rng_key, hmc_key, vi_key = random.split(rng_key, 3)
mcmc = run_hmc(logistic_regression, X, y, hmc_key)
svi_result, var_dist = run_vi(logistic_regression, X, y, vi_key)

# Get posterior samples
hmc_samples = mcmc.get_samples()
vi_samples = var_dist.sample_posterior(vi_key, svi_result.params, sample_shape=(2000,))
vi_posterior = var_dist.get_posterior(svi_result.params)

plot_predictions(hmc_samples, vi_samples, X, y)
plot_posterior(hmc_samples, vi_posterior, true_weights, true_bias)

plt.show()
