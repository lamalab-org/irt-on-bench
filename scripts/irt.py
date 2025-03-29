import pymc as pm
import numpy as np
import arviz as az


def fit_2pl_pymc(response_matrix, n_samples=2000, tune=2000):
    n_participants, n_items = response_matrix.shape

    with pm.Model() as irt_2pl:
        # 1. Standardize input data if not already done
        # response_matrix = (response_matrix - np.mean(response_matrix)) / np.std(response_matrix)

        # 2. Abilities with stronger regularization
        abilities = pm.Normal(
            "abilities",
            mu=0,
            sigma=0.5,  # Tighter prior
            shape=n_participants,
        )

        # 3. Simpler discrimination parameterization
        # Constrain to reasonable range [0.5, 2.5]
        discriminations = pm.TruncatedNormal(
            "discriminations", mu=1.0, sigma=0.3, lower=0.5, upper=2.5, shape=n_items
        )

        # 4. Difficulties with tighter bounds
        difficulties = pm.TruncatedNormal(
            "difficulties", mu=0, sigma=0.5, lower=-2, upper=2, shape=n_items
        )

        # 5. More stable computation with scaling
        scaled_abilities = abilities[:, None] / 2.0  # Scale down
        scaled_difficulties = difficulties[None, :] / 2.0
        scaled_disc = discriminations[None, :] / 2.0

        # Compute logit with scaled parameters
        logit_p = scaled_disc * (scaled_abilities - scaled_difficulties)

        # 6. Likelihood
        responses = pm.Bernoulli("responses", logit_p=logit_p, observed=response_matrix)

        # 7. Improved sampling settings
        trace = pm.sample(
            n_samples,
            tune=tune,
            chains=4,
            target_accept=0.99,
            init="jitter+adapt_diag",  # Different initializer
            return_inferencedata=True,
            cores=1,
        )  # Single core for better stability

    return trace


# Modern diagnostics using ArviZ
def check_model_diagnostics(trace):
    """Comprehensive model diagnostics"""
    diagnostics = {
        "r_hat": az.rhat(trace),
        "ess": az.ess(trace),
        "mcse": az.mcse(trace),
        "divergences": az.summary(trace)["diverging"].sum(),
    }

    # Check for convergence issues
    has_issues = (
        (diagnostics["r_hat"] > 1.01).any()
        or (diagnostics["ess"] < 400).any()
        or diagnostics["divergences"] > 0
    )

    if has_issues:
        print("Warning: Potential convergence issues detected")

    return diagnostics


def get_parameter_estimates(trace):
    estimates = {
        "abilities": {
            "mean": trace.posterior["abilities"].mean(dim=["chain", "draw"]).values,
            "std": trace.posterior["abilities"].std(dim=["chain", "draw"]).values,
            "hdi": az.hdi(trace, var_names=["abilities"]),
        },
        "discriminations": {
            "mean": trace.posterior["discriminations"]
            .mean(dim=["chain", "draw"])
            .values,
            "std": trace.posterior["discriminations"].std(dim=["chain", "draw"]).values,
            "hdi": az.hdi(trace, var_names=["discriminations"]),
        },
        "difficulties": {
            "mean": trace.posterior["difficulties"].mean(dim=["chain", "draw"]).values,
            "std": trace.posterior["difficulties"].std(dim=["chain", "draw"]).values,
            "hdi": az.hdi(trace, var_names=["difficulties"]),
        },
    }
    return estimates
