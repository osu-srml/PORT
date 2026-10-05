"""
Loss factory for dynamically selecting loss functions based on Hydra configuration.

This module provides a factory pattern for loss function selection, allowing
easy addition of new loss functions without modifying the trainer code.
"""

from typing import Dict, Tuple
from omegaconf import DictConfig
import torch

from .loss import (
    port_loss,
    concatenated_forward,
    compute_sft_loss_and_metrics,
    gather_and_compute_reward_metrics,
    gather_and_compute_kl_metrics,
    gather_rejected_logps_metrics,
)


# =============================================================================
# Registry: Maps loss names to their underlying loss computation functions
# =============================================================================

LOSS_FUNCTIONS = {
    "port": port_loss,
}

# Losses that require KL divergence metrics
LOSSES_WITH_KL = {"port"}


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_loss_params(loss_name: str, config: DictConfig, forward_outputs: tuple) -> dict:
    """Extract parameters needed for a specific loss function.

    Args:
        loss_name: Name of the loss function
        config: Loss configuration
        forward_outputs: Tuple of outputs from concatenated_forward

    Returns:
        Dictionary of parameters to pass to the loss function
    """
    (chosen_logps_margin, rejected_logps_margin,
     chosen_position_kl, rejected_position_kl,
     policy_chosen_logps, policy_rejected_logps,
     reference_chosen_logps, reference_rejected_logps) = forward_outputs

    params = {
        "policy_chosen_logps": policy_chosen_logps,
        "policy_rejected_logps": policy_rejected_logps,
        "reference_chosen_logps": reference_chosen_logps,
        "reference_rejected_logps": reference_rejected_logps,
        "beta": config.beta,
    }

    if loss_name in ["dpo", "ipo"]:
        return params

    params.update({
        "chosen_logps_margin": chosen_logps_margin,
        "rejected_logps_margin": rejected_logps_margin,
        "chosen_position_kl": chosen_position_kl,
        "rejected_position_kl": rejected_position_kl,
    })

    if loss_name in ["tdpo", "focal_tdpo", "PORT", "tdpo_cdf"]:
        params["alpha"] = config.alpha
        params["if_tdpo2"] = config.if_tdpo2

    if loss_name == "focal_tdpo":
        params["gamma"] = config.gamma

    if loss_name in ["PORT", "tdpo_cdf"]:
        params.update({
            "ranking_method": config.ranking_method,
            "weight_power": config.weight_power,
            "min_weight_ratio": config.min_weight_ratio,
            "N": config.N,
        })

    if loss_name in ["tdpo", "PORT", "tdpo_cdf"]:
        params["chosen_policy_logps"] = params.pop("policy_chosen_logps")
        params["rejected_policy_logps"] = params.pop("policy_rejected_logps")
        params["chosen_reference_logps"] = params.pop("reference_chosen_logps")
        params["rejected_reference_logps"] = params.pop("reference_rejected_logps")

    return params


# =============================================================================
# Main Loss Computation Function
# =============================================================================

def compute_loss_and_metrics(
    loss_name: str,
    config: DictConfig,
    policy,
    reference_model,
    batch,
    rank: int,
    world_size: int,
    train_test: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Unified interface for computing loss and metrics for any loss function.

    Args:
        loss_name: Name of the loss function (e.g., "dpo", "tdpo", "PORT")
        config: Loss configuration from Hydra
        policy: Policy model
        reference_model: Reference model (can be None for SFT)
        batch: Input batch
        rank: Current device rank
        world_size: Total number of devices
        train_test: "train" or "eval"

    Returns:
        Tuple of (losses, policy_chosen_logps, metrics)

    Raises:
        ValueError: If loss_name is not recognized
    """
    if loss_name == "sft":
        return compute_sft_loss_and_metrics(policy, batch)

    if loss_name not in LOSS_FUNCTIONS:
        available = ", ".join(["sft"] + list(LOSS_FUNCTIONS.keys()))
        raise ValueError(
            f"Unknown loss function: {loss_name}. "
            f"Available options: {available}"
        )

    forward_outputs = concatenated_forward(policy, reference_model, batch)

    (_, _,
     chosen_position_kl, rejected_position_kl,
     policy_chosen_logps, policy_rejected_logps,
     _, _) = forward_outputs

    loss_params = _extract_loss_params(loss_name, config, forward_outputs)

    loss_fn = LOSS_FUNCTIONS[loss_name]
    losses, chosen_rewards, rejected_rewards = loss_fn(**loss_params)

    metrics = {}

    metrics.update(gather_and_compute_reward_metrics(
        chosen_rewards, rejected_rewards, rank, world_size, train_test
    ))

    if loss_name in LOSSES_WITH_KL:
        metrics.update(gather_and_compute_kl_metrics(
            chosen_position_kl, rejected_position_kl, rank, world_size, train_test
        ))

    metrics.update(gather_rejected_logps_metrics(
        policy_rejected_logps, rank, world_size, train_test
    ))

    return losses, policy_chosen_logps, metrics


def requires_reference_model(loss_name: str) -> bool:
    """Check if a loss function requires a reference model.

    Args:
        loss_name: Name of the loss function

    Returns:
        True if the loss requires a reference model, False otherwise
    """
    return loss_name != "sft"


def get_available_losses() -> list:
    """Get list of all available loss functions."""
    return ["sft"] + list(LOSS_FUNCTIONS.keys())
