"""
Loss functions and utilities for preference learning.

This module contains:
1. Basic utility functions (data processing, log probability computation)
2. Forward pass functions (model inference with chosen/rejected pairs)
3. Core loss functions (PORT, etc.)
4. Metrics computation functions
5. High-level interface functions
"""

from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spawn import pad_to_length, all_gather_if_needed

SUPPORTED_PREFERENCE_LOSSES = ["dpo", "ipo", "tdpo", "focal_tdpo", "PORT", "tdpo_loss_cdf"]


# =============================================================================
# Section 1: Basic Utility Functions
# =============================================================================

def concatenated_inputs(
    batch: Dict[str, Union[List, torch.LongTensor]],
) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and
            'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

    Returns:
        A dictionary containing the concatenated inputs of shape (2 * batch_size, sequence_length).
    """
    max_length = max(
        batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1]
    )
    concatenated_batch = {}
    for k in batch:
        if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(
                batch[k], max_length, pad_value=pad_value
            )
    for k in batch:
        if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("rejected", "concatenated")
            concatenated_batch[concatenated_key] = torch.cat(
                (
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ),
                dim=0,
            )
    return concatenated_batch


def _get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of
            -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100

    labels[labels == -100] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def _tdpo_get_batch_logps(
    logits: torch.FloatTensor,
    reference_logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
):
    """Compute the KL divergence and log probabilities for TDPO.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        reference_logits: Logits of the reference model (unnormalized).
            Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of
            -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token.

    Returns:
        Several tensors of shape (batch_size,) containing:
        - logps_margin: Difference between policy and reference log probabilities
        - position_kl: KL divergence at each position
        - policy_logps: Policy model log probabilities
        - reference_logps: Reference model log probabilities
    """
    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = labels != -100

    labels[labels == -100] = 0

    vocab_logps = logits.log_softmax(-1)

    reference_vocab_logps = reference_logits.log_softmax(-1)
    reference_vocab_ps = reference_vocab_logps.exp()

    kl_diff = reference_vocab_logps - vocab_logps
    kl_diff = torch.clamp(kl_diff, min=-50, max=50)
    per_position_kl = (reference_vocab_ps * kl_diff).sum(-1)
    per_position_kl = torch.clamp(per_position_kl, min=-50, max=50)

    per_token_logps = torch.gather(
        vocab_logps, dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)
    per_reference_token_logps = torch.gather(
        reference_vocab_logps, dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    logps_margin = per_token_logps - per_reference_token_logps

    if average_log_prob:
        return (
            (logps_margin * loss_mask).sum(-1) / loss_mask.sum(-1),
            (per_position_kl * loss_mask).sum(-1) / loss_mask.sum(-1),
            (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1),
            (per_reference_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1),
        )
    else:
        return (
            (logps_margin * loss_mask).sum(-1),
            (per_position_kl * loss_mask).sum(-1),
            (per_token_logps * loss_mask).sum(-1),
            (per_reference_token_logps * loss_mask).sum(-1),
        )


def compute_empirical_cdf(
    target_values: torch.FloatTensor,
    reference_values: torch.FloatTensor
) -> torch.FloatTensor:
    """Compute empirical cumulative distribution function (CDF) values.

    This function computes the empirical CDF F(x) = P(X <= x) where X is distributed
    according to the reference_values. For each target value, it calculates the
    proportion of reference values that are less than or equal to that target value.

    Args:
        target_values: Values for which to compute CDF estimates. Shape: (n,)
        reference_values: Reference distribution samples used to estimate the CDF. Shape: (m,)

    Returns:
        torch.FloatTensor: Empirical CDF values for each target value. Shape: (n,).
    """
    cdf_values = torch.zeros_like(target_values)

    for i in range(target_values.size(0)):
        count = (reference_values <= target_values[i]).float().sum()
        cdf_values[i] = count / reference_values.size(0)

    return cdf_values


# =============================================================================
# Section 2: Forward Pass Functions
# =============================================================================

def concatenated_forward(
    model: nn.Module,
    reference_model: nn.Module,
    batch: Dict[str, Union[List, torch.LongTensor]],
):
    """Run forward pass for both policy and reference models on concatenated inputs.

    This function concatenates chosen and rejected inputs to perform only one forward pass
    for each model, which is more efficient especially for FSDP.

    Args:
        model: Policy model
        reference_model: Reference model
        batch: Input batch containing chosen and rejected pairs

    Returns:
        Tuple containing log prob margins, position KL, policy logps, and reference logps
        for both chosen and rejected responses.
    """
    concatenated_batch = concatenated_inputs(batch)
    all_logits = model(
        concatenated_batch["concatenated_input_ids"],
        attention_mask=concatenated_batch["concatenated_attention_mask"],
    ).logits.to(torch.float32)

    with torch.no_grad():
        reference_all_logits = reference_model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits.to(torch.float32)

    all_logps_margin, all_position_kl, all_logps, reference_all_logps = (
        _tdpo_get_batch_logps(
            all_logits,
            reference_all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
        )
    )

    batch_size = batch["chosen_input_ids"].shape[0]
    chosen_logps_margin = all_logps_margin[:batch_size]
    rejected_logps_margin = all_logps_margin[batch_size:]
    chosen_position_kl = all_position_kl[:batch_size]
    rejected_position_kl = all_position_kl[batch_size:]
    chosen_reference_logps = reference_all_logps[:batch_size]
    rejected_reference_logps = reference_all_logps[batch_size:]
    chosen_logps = all_logps[:batch_size]
    rejected_logps = all_logps[batch_size:]

    return (
        chosen_logps_margin,
        rejected_logps_margin,
        chosen_position_kl,
        rejected_position_kl,
        chosen_logps,
        rejected_logps,
        chosen_reference_logps,
        rejected_reference_logps,
    )


# =============================================================================
# Section 3: Core Loss Functions
# =============================================================================


def port_loss(
    chosen_logps_margin: torch.FloatTensor,
    rejected_logps_margin: torch.FloatTensor,
    chosen_position_kl: torch.FloatTensor,
    rejected_position_kl: torch.FloatTensor,
    chosen_policy_logps: torch.FloatTensor,
    rejected_policy_logps: torch.FloatTensor,
    chosen_reference_logps: torch.FloatTensor,
    rejected_reference_logps: torch.FloatTensor,
    beta: float,
    alpha: float = 0.5,
    if_tdpo2: bool = True,
    ranking_method: str = "vbon_difference",
    weight_power: float = 1.0,
    min_weight_ratio: float = 0.1,
    N: int = None,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the PORT (Ranking-based DPO) loss with sample weighting.

    Args:
        chosen_logps_margin: Log prob difference for chosen. Shape: (batch_size,)
        rejected_logps_margin: Log prob difference for rejected. Shape: (batch_size,)
        chosen_position_kl: KL divergence for chosen. Shape: (batch_size,)
        rejected_position_kl: KL divergence for rejected. Shape: (batch_size,)
        chosen_policy_logps: Policy log probs for chosen
        rejected_policy_logps: Policy log probs for rejected
        chosen_reference_logps: Reference log probs for chosen
        rejected_reference_logps: Reference log probs for rejected
        beta: Temperature parameter
        alpha: Temperature parameter for KL divergence
        if_tdpo2: If True, use TDPO2; if False, use TDPO1
        ranking_method: Method for ranking ("loss_based", "vbon_difference", "vbon_exponential")
        weight_power: Power to apply to ranking weights
        min_weight_ratio: Minimum weight ratio for least important sample
        N: Virtual Best-of-N parameter (if None, uses batch_size)

    Returns:
        Tuple of (weighted_losses, chosen_rewards, rejected_rewards)
    """
    chosen_values = chosen_logps_margin + chosen_position_kl
    rejected_values = rejected_logps_margin + rejected_position_kl
    chosen_rejected_logps_margin = chosen_logps_margin - rejected_logps_margin

    if not if_tdpo2:
        logits = chosen_rejected_logps_margin - (
            rejected_position_kl - chosen_position_kl
        )  # TDPO1
    else:
        logits = chosen_rejected_logps_margin - alpha * (
            rejected_position_kl - chosen_position_kl.detach()
        )  # TDPO2

    clamped_logits = torch.clamp(logits, min=-20, max=20)
    raw_losses = -F.logsigmoid(beta * clamped_logits)

    if ranking_method == "loss_based":
        ranking_values = raw_losses

    elif ranking_method.startswith("vbon_"):
        if N is None:
            N = chosen_values.size(0)

        chosen_rewards = beta * chosen_values.detach()
        rejected_rewards = beta * rejected_values.detach()

        all_rewards = torch.cat([chosen_rewards, rejected_rewards])

        chosen_cdf = compute_empirical_cdf(chosen_rewards, all_rewards)
        rejected_cdf = compute_empirical_cdf(rejected_rewards, all_rewards)

        if ranking_method == "vbon_absolute_difference":
            m = 0.3
            n = 1.5
            ranking_values = n * torch.abs(torch.exp(m * (chosen_cdf - rejected_cdf)))

        else:
            raise ValueError(f"Unknown vBoN ranking method: {ranking_method}")

        batch_size = chosen_values.size(0)
        if N > batch_size:
            scale_factor = (N / batch_size) ** 0.5
            ranking_values = ranking_values * scale_factor

    else:
        raise ValueError(f"Unknown ranking method: {ranking_method}")

    if ranking_method.startswith("vbon_"):
        ranking_weights = ranking_values
    else:
        batch_size = ranking_values.size(0)
        sorted_indices = torch.argsort(ranking_values, descending=True)

        ranks = torch.zeros_like(ranking_values)
        ranks[sorted_indices] = torch.arange(
            1, batch_size + 1, dtype=torch.float, device=ranking_values.device
        )

        ranking_weights = (batch_size + 1 - ranks) ** weight_power

    if ranking_weights.sum() > 0:
        max_weight = ranking_weights.max()
        min_weight = max_weight * min_weight_ratio
        ranking_weights = torch.clamp(ranking_weights, min=min_weight)
        ranking_weights = ranking_weights / ranking_weights.mean()
    else:
        ranking_weights = torch.ones_like(ranking_weights)

    weighted_losses = raw_losses * ranking_weights
    chosen_rewards = beta * chosen_values.detach()
    rejected_rewards = beta * rejected_values.detach()

    return weighted_losses, chosen_rewards, rejected_rewards




# =============================================================================
# Section 4: Metrics Computation Functions
# =============================================================================

def gather_and_compute_reward_metrics(
    chosen_rewards,
    rejected_rewards,
    rank,
    world_size,
    train_test
):
    """Gather rewards across devices and compute reward metrics."""
    reward_accuracies = (chosen_rewards > rejected_rewards).float()

    chosen_rewards = all_gather_if_needed(chosen_rewards, rank, world_size)
    rejected_rewards = all_gather_if_needed(rejected_rewards, rank, world_size)
    reward_accuracies = all_gather_if_needed(reward_accuracies, rank, world_size)

    metrics = {
        f"rewards_{train_test}/chosen": chosen_rewards.cpu().numpy().tolist(),
        f"rewards_{train_test}/rejected": rejected_rewards.cpu().numpy().tolist(),
        f"rewards_{train_test}/accuracies": reward_accuracies.cpu().numpy().tolist(),
        f"rewards_{train_test}/margins": (chosen_rewards - rejected_rewards).cpu().numpy().tolist(),
    }

    return metrics


def gather_and_compute_kl_metrics(
    chosen_position_kl,
    rejected_position_kl,
    rank,
    world_size,
    train_test
):
    """Gather KL divergences across devices and compute KL metrics."""
    all_device_chosen_position_kl = all_gather_if_needed(
        chosen_position_kl.detach(), rank, world_size
    )
    all_device_rejected_position_kl = all_gather_if_needed(
        rejected_position_kl.detach(), rank, world_size
    )

    metrics = {
        f"kl_{train_test}/chosen": all_device_chosen_position_kl.cpu().numpy().tolist(),
        f"kl_{train_test}/rejected": all_device_rejected_position_kl.cpu().numpy().tolist(),
        f"kl_{train_test}/margin": (
            all_device_chosen_position_kl - all_device_rejected_position_kl
        ).cpu().numpy().tolist(),
    }

    return metrics


def gather_rejected_logps_metrics(
    policy_rejected_logps,
    rank,
    world_size,
    train_test
):
    """Gather rejected log probabilities and return metrics."""
    policy_rejected_logps = all_gather_if_needed(
        policy_rejected_logps.detach(), rank, world_size
    )

    return {
        f"logps_{train_test}/rejected": policy_rejected_logps.cpu().numpy().tolist()
    }


# =============================================================================
# Section 5: High-Level Interface Functions
# =============================================================================

def compute_sft_loss_and_metrics(policy, batch):
    """Compute SFT (Supervised Fine-Tuning) loss and return necessary outputs.

    Args:
        policy: Policy model
        batch: Input batch

    Returns:
        Tuple of (losses, policy_chosen_logps, metrics)
    """
    policy_chosen_logits = policy(
        batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]
    ).logits.to(torch.float32)
    policy_chosen_logps = _get_batch_logps(
        policy_chosen_logits, batch["chosen_labels"], average_log_prob=False
    )

    losses = -policy_chosen_logps
    metrics = {}

    return losses, policy_chosen_logps, metrics


