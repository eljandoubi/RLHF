from typing import Callable

import torch


def compute_group_normalized_rewards(
    reward_fnn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized by the group size.
    Args:
    reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against
    the ground truths, producing a dict with keys "reward", "format_reward", and
    "answer_reward".
    rollout_responses: list[str] Rollouts from the policy. The length of this list is
    rollout_batch_size = n_prompts_per_rollout_batch * group_size.
    repeated_ground_truths: list[str] The ground truths for the examples. The length of this
    list is rollout_batch_size, because the ground truth for each example is repeated
    group_size times.
    group_size: int Number of responses per question (group).
    advantage_eps: float Small constant to avoid division by zero in normalization.
    normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise
    subtract only the group mean.
    Returns:
    tuple[torch.Tensor, torch.Tensor, dict[str, float]].
    advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout
    response.
    22
    raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout
    response.
    metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """
    outputs = list(map(reward_fnn, rollout_responses, repeated_ground_truths))
    raw_rewards = torch.tensor([output["reward"] for output in outputs])
    rewards = raw_rewards.view(-1, group_size)
    mean_rewards = rewards.mean(dim=1, keepdim=True)
    if normalize_by_std:
        std_rewards = rewards.std(dim=1, keepdim=True)
        advantages = (rewards - mean_rewards) / (std_rewards + advantage_eps)
    else:
        advantages = rewards - mean_rewards
    return (advantages.view(-1), raw_rewards, 
            {"mean_reward": mean_rewards.mean().item(),
             "std_reward": rewards.std().item(),
             "max_reward": rewards.max().item(),
             "min_reward": rewards.min().item()})

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    ) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either
    the raw reward or an already-normalized advantage.
    Args:
    raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar
    reward/advantage for each rollout response.
    policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for
    each token.
    Returns:
    torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss (to
    be aggregated across the batch and sequence dimensions in the training loop).
    """
    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
    advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
    policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log
    probs from the policy being trained.
    old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs
    from the old policy.
    cliprange: float Clip parameter ε (e.g. 0.2).
    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
    loss torch.Tensor of shape (batch_size, sequence_length), the per-token clipped
    loss.
    metadata dict containing whatever you want to log. We suggest logging whether each
    token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of
    the min was lower than the LHS.
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    loss1 = advantages * ratio
    loss2 = advantages * clipped_ratio
    loss = -torch.minimum(loss1, loss2)
    with torch.inference_mode():
        metadata = {"clipped": loss1 >= loss2}
    return loss, metadata

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    ) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those elements where
    mask == 1.
    Args:
    tensor: torch.Tensor The data to be averaged.
    mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
    dim: int | None Dimension over which to average. If None, compute the mean over all
    masked elements.
    Returns:
    torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
    """
    masked_sum = (tensor * mask.to(tensor.dtype)).sum(dim=dim)
    count = mask.sum(dim=dim)
    return masked_sum / count