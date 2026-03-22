from argparse import Namespace
from typing import Callable, Literal

import torch
from sft import get_optimizer, init_vllm, prepare_data
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


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

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.
    Args:
    policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
    policy being trained.
    loss_type One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
    raw_rewards Required if loss_type == "no_baseline"; shape (batch_size, 1).
    advantages Required for "reinforce_with_baseline" and "grpo_clip"; shape
    (batch_size, 1).
    old_log_probs Required for "grpo_clip"; shape (batch_size, sequence_length).
    cliprange Required for "grpo_clip"; scalar ε used for clipping.
    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
    loss (batch_size, sequence_length), per-token loss.
    metadata dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip)
    """
    metadata = {}
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards is required for no-baseline loss"
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards, policy_log_probs=policy_log_probs
        )
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages are required for reinforce with baseline loss"
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages, policy_log_probs=policy_log_probs
        )
    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages are required for GRPO-Clip loss"
        assert old_log_probs is not None, "old_log_probs are required for GRPO-Clip loss"
        assert cliprange is not None, "cliprange is required for GRPO-Clip loss"
        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
    else:
        raise ValueError(f"Invalid loss_type: {loss_type}")

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


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.
    Args:
    policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
    policy being trained.
    response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
    prompt/padding.
    gradient_accumulation_steps Number of microbatches per optimizer step.
    loss_type One of "no_baseline", "reinforce_with_baseline", "grpo_clip".
    raw_rewards Needed when loss_type == "no_baseline"; shape (batch_size, 1).
    advantages Needed when loss_type != "no_baseline"; shape (batch_size, 1).
    old_log_probs Required for GRPO-Clip; shape (batch_size, sequence_length).
    cliprange Clip parameter ε for GRPO-Clip.
    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
    loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
    this so we can log it.
    metadata Dict with metadata from the underlying loss call, and any other statistics you
    might want to log.
    """
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    loss = masked_mean(loss, response_mask, dim=1).mean() / gradient_accumulation_steps
    loss.backward()
    return loss.detach(), metadata

def grpo_training(args: Namespace):
    """
    Main training loop for GRPO. You can structure this however you want; we suggest
    following the general structure of the SFT training loop in sft.py, but replacing the loss
    computation with calls to the above functions.
    """
    policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        args.policy_model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(args.policy_device)
    optimizer_cls = get_optimizer(args.optimizer)
    optimizer: torch.optim.Optimizer = optimizer_cls(
        policy.parameters(), lr=args.learning_rate
    )
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    ref_model = init_vllm(
        args.ref_model_id,
        device=args.vllm_device,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    dataset = prepare_data(
        args.dataset_name,
        test_size=args.test_size,
        seed=args.seed,
        num_proc=args.num_proc,
    )
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    policy.train()