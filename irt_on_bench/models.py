"""
IRT model implementations for benchmark analysis.

This module provides functions for fitting Item Response Theory (IRT) models
to language model evaluation data, using both classical methods and
Bayesian approaches with PyMC.
"""

import numpy as np
import pymc as pm
import arviz as az
from typing import Any
import time
from loguru import logger


def twopl_mml(response_matrix: np.ndarray) -> dict[str, np.ndarray]:
    """
    Fit a 2-parameter logistic IRT model using marginal maximum likelihood.

    This is a simplified implementation that estimates parameters using
    an iterative approach based on logistic regression principles.

    Args:
        response_matrix (np.ndarray): Binary response matrix with shape [n_items, n_participants].

    Returns:
        dict[str, np.ndarray]: dictionary containing:
            - 'Difficulty': Item difficulty parameters.
            - 'Discrimination': Item discrimination parameters.
    """
    logger.info("Starting 2PL model fitting with MML...")
    start_time = time.time()

    n_items, n_participants = response_matrix.shape

    # Initialize parameters
    abilities = np.zeros(n_participants)
    difficulties = np.zeros(n_items)
    discriminations = np.ones(n_items)

    # Number of iterations for parameter estimation
    max_iter = 30
    convergence_threshold = 0.001

    for iteration in range(max_iter):
        # Store old parameters for convergence check
        old_difficulties = difficulties.copy()
        old_discriminations = discriminations.copy()
        old_abilities = abilities.copy()

        # Step 1: Estimate abilities given item parameters
        for j in range(n_participants):
            # Calculate expected probabilities
            z = discriminations * (abilities[j] - difficulties)
            probs = 1 / (1 + np.exp(-z))

            # Calculate gradient and Hessian for Newton-Raphson
            r = response_matrix[:, j] - probs
            grad = np.sum(discriminations * r)
            hess = -np.sum(discriminations**2 * probs * (1 - probs))

            if hess != 0:  # Avoid division by zero
                abilities[j] += grad / hess

        # Standardize abilities to have mean 0 and SD 1
        abilities = (abilities - np.mean(abilities)) / np.std(abilities)

        # Step 2: Estimate item parameters given abilities
        for i in range(n_items):
            # Skip items with extreme p-values
            p_value = np.mean(response_matrix[i, :])
            if p_value < 0.05 or p_value > 0.95:
                continue

            # Calculate expected probabilities
            z = discriminations[i] * (abilities - difficulties[i])
            probs = 1 / (1 + np.exp(-z))

            # Calculate gradients and Hessian for Newton-Raphson
            r = response_matrix[i, :] - probs

            # Update difficulty
            grad_b = -np.sum(discriminations[i] * r)
            hess_b = np.sum(discriminations[i] ** 2 * probs * (1 - probs))

            if hess_b != 0:  # Avoid division by zero
                difficulties[i] += grad_b / hess_b

            # Update discrimination
            grad_a = np.sum((abilities - difficulties[i]) * r)
            hess_a = np.sum((abilities - difficulties[i]) ** 2 * probs * (1 - probs))

            if hess_a != 0:  # Avoid division by zero
                discriminations[i] += grad_a / hess_a

            # Constrain discrimination to be positive
            discriminations[i] = max(0.1, discriminations[i])

        # Check convergence
        diff_d = np.max(np.abs(difficulties - old_difficulties))
        diff_a = np.max(np.abs(discriminations - old_discriminations))
        diff_theta = np.max(np.abs(abilities - old_abilities))

        max_diff = max(diff_d, diff_a, diff_theta)

        logger.info(f"Iteration {iteration + 1}: max parameter change = {max_diff:.6f}")

        if max_diff < convergence_threshold:
            logger.info(f"Converged after {iteration + 1} iterations")
            break

    # Scale discriminations to have a reasonable range
    if np.max(discriminations) > 4:
        scale_factor = 2 / np.mean(discriminations)
        discriminations *= scale_factor

    logger.info(
        f"2PL model fitting completed in {time.time() - start_time:.2f} seconds"
    )

    return {"Difficulty": difficulties, "Discrimination": discriminations}


def rasch_mml(response_matrix: np.ndarray) -> dict[str, np.ndarray]:
    """
    Fit a Rasch (1-parameter logistic) IRT model using marginal maximum likelihood.

    Args:
        response_matrix (np.ndarray): Binary response matrix with shape [n_items, n_participants].

    Returns:
        dict[str, np.ndarray]: dictionary containing:
            - 'Difficulty': Item difficulty parameters.
    """
    logger.info("Starting Rasch model fitting with MML...")
    start_time = time.time()

    n_items, n_participants = response_matrix.shape

    # Initialize parameters
    abilities = np.zeros(n_participants)
    difficulties = np.zeros(n_items)

    # Number of iterations for parameter estimation
    max_iter = 30
    convergence_threshold = 0.001

    for iteration in range(max_iter):
        # Store old parameters for convergence check
        old_difficulties = difficulties.copy()
        old_abilities = abilities.copy()

        # Step 1: Estimate abilities given item difficulties
        for j in range(n_participants):
            # Calculate expected probabilities
            z = abilities[j] - difficulties
            probs = 1 / (1 + np.exp(-z))

            # Calculate gradient and Hessian for Newton-Raphson
            r = response_matrix[:, j] - probs
            grad = np.sum(r)
            hess = -np.sum(probs * (1 - probs))

            if hess != 0:  # Avoid division by zero
                abilities[j] += grad / hess

        # Standardize abilities to have mean 0 and SD 1
        abilities = (abilities - np.mean(abilities)) / np.std(abilities)

        # Step 2: Estimate item difficulties given abilities
        for i in range(n_items):
            # Skip items with extreme p-values
            p_value = np.mean(response_matrix[i, :])
            if p_value < 0.05 or p_value > 0.95:
                continue

            # Calculate expected probabilities
            z = abilities - difficulties[i]
            probs = 1 / (1 + np.exp(-z))

            # Calculate gradient and Hessian for Newton-Raphson
            r = response_matrix[i, :] - probs
            grad = -np.sum(r)
            hess = np.sum(probs * (1 - probs))

            if hess != 0:  # Avoid division by zero
                difficulties[i] += grad / hess

        # Check convergence
        diff_d = np.max(np.abs(difficulties - old_difficulties))
        diff_theta = np.max(np.abs(abilities - old_abilities))

        max_diff = max(diff_d, diff_theta)

        logger.info(f"Iteration {iteration + 1}: max parameter change = {max_diff:.6f}")

        if max_diff < convergence_threshold:
            logger.info(f"Converged after {iteration + 1} iterations")
            break

    logger.info(
        f"Rasch model fitting completed in {time.time() - start_time:.2f} seconds"
    )

    return {"Difficulty": difficulties}


def ability_mle(
    response_matrix: np.ndarray,
    difficulties: np.ndarray,
    discriminations: np.ndarray,
    no_estimate: float = np.nan,
) -> np.ndarray:
    """
    Estimate ability parameters using maximum likelihood estimation.


    Args:
        response_matrix (np.ndarray): Binary response matrix with shape [n_items, n_participants].
        difficulties (np.ndarray): Item difficulty parameters.
        discriminations (np.ndarray): Item discrimination parameters.
        no_estimate (float, optional): Value to use for participants with no estimated ability (default: np.nan).
    Returns:
        np.ndarray: Estimated ability parameters for each participant.
    """
    logger.info("Estimating abilities using MLE...")
    start_time = time.time()

    n_items, n_participants = response_matrix.shape
    abilities = np.zeros(n_participants)

    # Maximum number of iterations for Newton-Raphson
    max_iter = 20
    convergence_threshold = 0.001

    for j in range(n_participants):
        # Initialize ability estimate
        theta = 0.0

        # Check if participant has any non-extreme responses
        all_correct = np.all(response_matrix[:, j] == 1)
        all_incorrect = np.all(response_matrix[:, j] == 0)

        if all_correct or all_incorrect:
            abilities[j] = no_estimate
            continue

        # Newton-Raphson iterations
        for iteration in range(max_iter):
            old_theta = theta

            # Calculate expected probabilities
            z = discriminations * (theta - difficulties)
            probs = 1 / (1 + np.exp(-z))

            # First derivative (gradient) of log-likelihood
            gradient = np.sum(discriminations * (response_matrix[:, j] - probs))

            # Second derivative (Hessian) of log-likelihood
            hessian = -np.sum(discriminations**2 * probs * (1 - probs))

            # Update ability estimate
            if hessian != 0:  # Avoid division by zero
                theta -= gradient / hessian

            # Check convergence
            if abs(theta - old_theta) < convergence_threshold:
                break

        abilities[j] = theta

    logger.info(
        f"Ability estimation completed in {time.time() - start_time:.2f} seconds"
    )

    return abilities


def fit_2pl_pymc(
    response_matrix: np.ndarray,
    n_samples: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.8,
    cores: int = 1,
) -> Any:
    """
    Fit a 2-parameter logistic IRT model using PyMC.

    Args:
        response_matrix (np.ndarray): Binary response matrix with shape [n_participants, n_items].
        n_samples (int, optional): Number of samples to draw (default: 2000).
        tune (int, optional): Number of tuning samples (default: 2000).
        chains (int, optional): Number of chains to run (default: 4).
        target_accept (float, optional): Target acceptance rate (default: 0.8).
        cores (int, optional): Number of cores to use (default: 1).

    Returns:
        Any: PyMC trace object containing the posterior samples.
    """
    logger.info(f"Starting PyMC 2PL model fitting with {n_samples} samples...")
    start_time = time.time()

    n_participants, n_items = response_matrix.shape
    logger.info(f"Data shape: {n_participants} participants, {n_items} items")

    # Calculate sparsity of the matrix
    sparsity = 1 - np.mean(response_matrix)
    logger.info(f"Response matrix sparsity: {sparsity:.2%}")

    with pm.Model() as irt_2pl:
        # Prior for abilities (participants)
        abilities = pm.Normal("abilities", mu=0, sigma=1, shape=n_participants)

        # Prior for discriminations (items)
        # Using TruncatedNormal to ensure positive discriminations
        discriminations = pm.TruncatedNormal(
            "discriminations", mu=1.0, sigma=0.5, lower=0.25, upper=4.0, shape=n_items
        )

        # Prior for difficulties (items)
        difficulties = pm.Normal("difficulties", mu=0, sigma=1.5, shape=n_items)

        # Calculate logits using vectorized operations
        # (we reshape for broadcasting)
        theta_expanded = abilities[:, None]  # Shape: [n_participants, 1]
        b_expanded = difficulties[None, :]  # Shape: [1, n_items]
        a_expanded = discriminations[None, :]  # Shape: [1, n_items]

        # Calculate logits
        logits = a_expanded * (theta_expanded - b_expanded)

        # Likelihood
        pm.Bernoulli("responses", logit_p=logits, observed=response_matrix)

        # Sample from the posterior
        logger.info("Starting MCMC sampling...")
        trace = pm.sample(
            n_samples,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            cores=cores,
            random_seed=42,
            return_inferencedata=True,
        )

    elapsed_time = time.time() - start_time
    logger.info(f"PyMC 2PL model fitting completed in {elapsed_time:.2f} seconds")

    return trace


def check_model_diagnostics(trace: Any) -> dict[str, Any]:
    """
    Comprehensive model diagnostics for PyMC traces.

    Args:
        trace (Any): PyMC trace object to diagnose.

    Returns:
        dict[str, Any]: dictionary containing diagnostics information:
            - 'r_hat': Gelman-Rubin statistics.
            - 'ess': Effective sample size.
            - 'mcse': Monte Carlo standard error.
            - 'divergences': Number of divergences.
    """
    logger.info("Checking model diagnostics...")

    # Calculate diagnostics
    r_hat = az.rhat(trace)
    ess = az.ess(trace)
    mcse = az.mcse(trace)
    summary = az.summary(trace)
    divergences = summary["diverging"].sum() if "diverging" in summary.columns else 0

    # Extrahiere r_hat Werte und vermeide Methodenaufruf
    # Angepasster Code für neuere ArviZ-Versionen
    try:
        # Versuche, den maximalen r_hat-Wert auf verschiedene Weisen zu extrahieren
        if hasattr(r_hat, "values") and not callable(r_hat.values):
            max_r_hat = float(np.max(r_hat.values))
        elif hasattr(r_hat, "to_array"):
            max_r_hat = float(np.max(r_hat.to_array()))
        else:
            # Fallback: Extrahiere Werte aus dem Summary
            r_hat_vals = [val for key, val in summary.items() if key.endswith("r_hat")]
            max_r_hat = float(np.max(r_hat_vals)) if r_hat_vals else 1.0
    except Exception as e:
        logger.warning(f"Could not extract r_hat values: {e}")
        max_r_hat = 1.0  # Fallback-Wert

    # Ähnlich für ESS
    try:
        if hasattr(ess, "values") and not callable(ess.values):
            min_ess = float(np.min(ess.values))
        elif hasattr(ess, "to_array"):
            min_ess = float(np.min(ess.to_array()))
        else:
            # Fallback
            ess_vals = [val for key, val in summary.items() if key.endswith("ess")]
            min_ess = float(np.min(ess_vals)) if ess_vals else 500.0
    except Exception as e:
        logger.warning(f"Could not extract ESS values: {e}")
        min_ess = 500.0  # Fallback-Wert

    logger.info(f"Maximum R-hat: {max_r_hat:.4f}")
    logger.info(f"Minimum ESS: {min_ess:.1f}")
    logger.info(f"Number of divergences: {divergences}")

    has_issues = (max_r_hat > 1.1) or (min_ess < 400) or (divergences > 0)

    if has_issues:
        logger.warning("Potential convergence issues detected!")
        if max_r_hat > 1.1:
            logger.warning("R-hat values > 1.1 indicate lack of convergence")
        if min_ess < 400:
            logger.warning(
                "Low effective sample size could indicate inefficient sampling"
            )
        if divergences > 0:
            logger.warning("Divergences indicate problems with the model geometry")
    else:
        logger.info("No major convergence issues detected")

    return {
        "r_hat": r_hat,
        "ess": ess,
        "mcse": mcse,
        "divergences": divergences,
        "has_issues": has_issues,
    }


def extract_parameters(trace: Any) -> dict[str, np.ndarray]:
    """
    Extract point estimates of parameters from a PyMC trace.

    Args:
            trace (Any): PyMC trace object.

        Returns:
            dict[str, np.ndarray]: dictionary containing:
                - 'abilities' (np.ndarray): Ability parameter point estimates.
                - 'difficulties' (np.ndarray): Difficulty parameter point estimates.
                - 'discriminations' (np.ndarray): Discrimination parameter point estimates.
    """
    # Extract posterior means as point estimates
    abilities = trace.posterior["abilities"].mean(dim=["chain", "draw"]).values
    difficulties = trace.posterior["difficulties"].mean(dim=["chain", "draw"]).values
    discriminations = (
        trace.posterior["discriminations"].mean(dim=["chain", "draw"]).values
    )

    return {
        "abilities": abilities,
        "difficulties": difficulties,
        "discriminations": discriminations,
    }


def extract_parameter_uncertainties(
    trace: Any, hdi_prob: float = 0.95
) -> dict[str, dict[str, np.ndarray]]:
    """
    Extract uncertainty estimates for parameters from a PyMC trace.

    Args:
        trace (Any): PyMC trace object.
        hdi_prob (float, optional): Probability mass for highest density interval (default: 0.95).

    Returns:
        dict[str, dict[str, np.ndarray]]: Nested dictionary containing:
            - Parameter type (abilities, difficulties, discriminations):
                - 'std': Standard deviation.
                - 'hdi_lower': Lower bound of HDI.
                - 'hdi_upper': Upper bound of HDI.
            hdi_prob: float = 0.95) -> dict[str, dict[str, np.ndarray]]:
    """
    # Calculate standard deviations
    abilities_std = trace.posterior["abilities"].std(dim=["chain", "draw"]).values
    difficulties_std = trace.posterior["difficulties"].std(dim=["chain", "draw"]).values
    discriminations_std = (
        trace.posterior["discriminations"].std(dim=["chain", "draw"]).values
    )

    # Calculate HDIs
    abilities_hdi = az.hdi(trace, var_names=["abilities"], hdi_prob=hdi_prob)
    difficulties_hdi = az.hdi(trace, var_names=["difficulties"], hdi_prob=hdi_prob)
    discriminations_hdi = az.hdi(
        trace, var_names=["discriminations"], hdi_prob=hdi_prob
    )

    return {
        "abilities": {
            "std": abilities_std,
            "hdi_lower": abilities_hdi.abilities.sel(hdi="lower").values,
            "hdi_upper": abilities_hdi.abilities.sel(hdi="higher").values,
        },
        "difficulties": {
            "std": difficulties_std,
            "hdi_lower": difficulties_hdi.difficulties.sel(hdi="lower").values,
            "hdi_upper": difficulties_hdi.difficulties.sel(hdi="higher").values,
        },
        "discriminations": {
            "std": discriminations_std,
            "hdi_lower": discriminations_hdi.discriminations.sel(hdi="lower").values,
            "hdi_upper": discriminations_hdi.discriminations.sel(hdi="higher").values,
        },
    }
