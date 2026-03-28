from argparse import ArgumentParser, Namespace
from functools import partial
from typing import Callable, Literal

import torch
from accelerate.utils.memory import clear_device_cache
from torch.nn import functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from vllm import RequestOutput, SamplingParams

import wandb
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.early_stopping import EarlyStopping
from cs336_alignment.parallel_mapper import ParallelMapper
from cs336_alignment.sft import (
    core_log_gen,
    get_optimizer,
    init_vllm,
    load_policy_into_vllm_instance,
    log_generations,
    prepare_data,
)
from cs336_alignment.summable_dict import SummableDict

r1_zero_reward_fn = partial(r1_zero_reward_fn, fast=False)


def compute_group_normalized_rewards(
    reward_fnn: ParallelMapper | Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
    processes: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized by the group size.
    Args:
    reward_fn: ParallelMapper Scores the rollout responses against
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
    processes: int | None Number of processes to use for parallelization. If None, defaults to the number of CPU cores.
    Returns:
    tuple[torch.Tensor, torch.Tensor, dict[str, float]].
    advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout
    response.
    raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout
    response.
    metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """
    if not isinstance(reward_fnn, ParallelMapper):
        reward_fnn = ParallelMapper(reward_fnn, processes)
    outputs = reward_fnn.map(rollout_responses, repeated_ground_truths)
    raw_rewards = torch.tensor([output["reward"] for output in outputs])
    format_rewards = torch.tensor([output["format_reward"] for output in outputs])
    answer_rewards = torch.tensor([output["answer_reward"] for output in outputs])
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
             "min_reward": rewards.min().item(),
             "mean_format_reward": format_rewards.mean().item(),
             "std_format_reward": format_rewards.std().item(),
             "max_format_reward": format_rewards.max().item(),
             "min_format_reward": format_rewards.min().item(),
             "std_answer_reward": answer_rewards.std().item(),
             "max_answer_reward": answer_rewards.max().item(),
             "min_answer_reward": answer_rewards.min().item(),
             "mean_answer_reward": answer_rewards.mean().item(),
            })

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
        metadata = {"clipped": (loss1.detach() >= loss2.detach()).float().mean()}
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

def get_rollout_logprobs(outputs:list[RequestOutput]) -> torch.Tensor:
    rollout_logprobs = [  ([0.0] *(len(ref_gen.prompt_token_ids)-1)
                           +[next(iter(pb.values())).logprob 
                                    for pb in out.logprobs])
                         for ref_gen in outputs 
                         for out in ref_gen.outputs]

    old_logprobs = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(m) for m in rollout_logprobs], 
            batch_first=True, padding_value=0.0
        )
    return old_logprobs

def prepare_mask(outputs:list[RequestOutput], device: torch.device) -> torch.Tensor:

    return torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([0]*(len(ref_gen.prompt_token_ids)-1) + [1]*len(out.token_ids)) 
             for ref_gen in outputs for out in ref_gen.outputs],
            batch_first=True, padding_value=0
        ).to(device=device,dtype=torch.bool)
    


def get_policy_log_probs(
    policy: PreTrainedModel,
    outputs:list[RequestOutput],
    pad_token_id: int,
    micro_batch_size: int
) -> torch.Tensor:
    """
    Get per-token log probabilities from the policy for the given inputs,
    optimized for a training loop.
    """

    log_probs_list = []
    for i in range(0, len(outputs), micro_batch_size):
        # Get logits for the micro-batch
        seq_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ref_gen.prompt_token_ids + list(out.token_ids)) 
             for ref_gen in outputs[i:i+micro_batch_size]
             for out in ref_gen.outputs], 
            batch_first=True, padding_value=pad_token_id
        ).to(device=policy.device,dtype=torch.long)

        model_output = policy(input_ids=seq_ids[:,:-1])
        logits_chunk = model_output.logits
    
        # --- Perform expensive operations on the small chunk ---
        log_probs_full_chunk = F.log_softmax(logits_chunk, dim=-1)
        
        # Get labels for the current chunk
        labels_chunk = seq_ids[:, 1:].unsqueeze(-1)
        
        # Gather the log probs for the actual tokens
        log_probs_chunk = log_probs_full_chunk.gather(dim=-1, index=labels_chunk)
        
        log_probs_list.append(log_probs_chunk.squeeze(-1))
    
    # Concatenate the final, small tensors
    all_log_probs = torch.cat(log_probs_list, dim=0)
    return all_log_probs

def grpo_training(args: Namespace):
    """
    Main training loop for GRPO. You can structure this however you want; we suggest
    following the general structure of the SFT training loop in sft.py, but replacing the loss
    computation with calls to the above functions.
    """
    assert args.loss_type in ["no_baseline", "reinforce_with_baseline", "grpo_clip"], f"Invalid loss type: {args.loss_type}"
    assert args.train_batch_size % args.gradient_accumulation_steps == 0, (
    "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    assert args.train_batch_size % args.group_size == 0, (
    "train_batch_size must be divisible by group_size"
    )
    assert args.train_batch_size >= args.group_size, (
    "train_batch_size must be greater than or equal to group_size"
    )

    policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        args.policy_model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(args.policy_device)
    optimizer_cls = get_optimizer(args.optimizer)
    optimizer: torch.optim.Optimizer = optimizer_cls(
        policy.parameters(), lr=args.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95)
    )
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    ref_model = init_vllm(
        args.ref_model_id,
        device=args.vllm_device,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    train_sampling_params = SamplingParams(
        temperature=args.sampling_temperature,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        logprobs=1 if args.loss_type == "grpo_clip" else None,  # We need old_log_probs for GRPO-Clip loss
        n=args.group_size,
    )

    eval_sampling_params = SamplingParams(
        temperature=args.sampling_temperature,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        logprobs=1,
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
    len_train_dataset = len(train_dataset)
    train_step_per_epoch = len_train_dataset // micro_train_batch_size + int(
        len_train_dataset % micro_train_batch_size > 0
    )
    len_eval_dataset = len(eval_dataset)
    eval_step_per_epoch = len_eval_dataset // args.eval_batch_size + int(
        len_eval_dataset % args.eval_batch_size > 0
    )
    total_steps = train_step_per_epoch * args.epochs
    progress_bar = tqdm(total=total_steps, desc="GRPO Training")
    mean_metadata = SummableDict()
    counter = 0
    parallel_reward_fn = ParallelMapper(r1_zero_reward_fn, processes=args.num_proc)
    eval_log_core = ParallelMapper(partial(core_log_gen,return_objects="only_sum"),
                                   processes=args.num_proc)
    parallel_log_core = ParallelMapper(partial(core_log_gen,return_objects="both"),
                                   processes=args.num_proc)
    

    early_stopper = EarlyStopping(
        metric_name=args.early_stopping_metric,
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        mode="max",
        min_steps=10,  # optional warmup
        smoothing_window=3,  # helps with RL noise
        output_dir=args.output_dir,
    )
    for i in range(args.epochs):
        for j, samples in enumerate(
            train_dataset.iter(batch_size=micro_train_batch_size)
        ):
            step = i * train_step_per_epoch + j + 1

            if step % args.ref_sync_steps == 0:
                load_policy_into_vllm_instance(policy, ref_model)
                clear_device_cache(garbage_collection=True)

            if step % args.eval_step == 0:
                tqdm.write(f"Evaluating at step {step}...")
                clear_device_cache(garbage_collection=True)
                avg_scores =  SummableDict()
                for eval_samples in tqdm(eval_dataset.iter(batch_size=args.eval_batch_size), desc="Evaluating",
                                         total=eval_step_per_epoch):
                    avg_scores += log_generations(
                            model=ref_model,
                            prompts=eval_samples["prompt"],
                            ground_truths=eval_samples["response"],
                            parallel_core=eval_log_core,
                            sampling_params=eval_sampling_params,
                            return_objects="only_sum",
                        )["only_sum"]

                num_samples = avg_scores.pop("num_samples", 1)
                avg_scores = avg_scores / num_samples
                tqdm.write("Average Scores:")
                for metric, score in avg_scores.items():
                    tqdm.write(f"  {metric}: {score:.4f}")
                wandb.log({f"eval/{k}": v for k, v in avg_scores.items()}, step=step)
                
                save_path = f"{args.output_dir}/checkpoint-{step}"
                policy.save_pretrained(save_path)
                tqdm.write(f"Saved checkpoint to {save_path}")
                
                stop, es_info = early_stopper.update(avg_scores, model=policy)

                tqdm.write(
                    f"[EarlyStopping] metric={es_info['smoothed_metric']:.4f}, "
                    f"best={es_info['best_metric']:.4f}, "
                    f"patience={es_info['patience_counter']}/{args.early_stopping_patience}"
                )

                wandb.log(
                    {
                        "early_stopping/metric": es_info["smoothed_metric"],
                        "early_stopping/best": es_info["best_metric"],
                        "early_stopping/patience": es_info["patience_counter"],
                    },
                    step=step,
                )

                if stop:
                    tqdm.write("Early stopping triggered.")
                    return

            prompts:list[str] = samples["prompt"]
            ground_truths:list[str] = samples["response"]
            repeated_ground_truths = [gt for gt in ground_truths for _ in range(args.group_size)]
            ref_outputs = ref_model.generate(
                prompts,
                sampling_params=train_sampling_params,
                use_tqdm=False,
            )
            rollout_responses = [out.text for ref_gen in ref_outputs for out in ref_gen.outputs]
            advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
                reward_fnn=parallel_reward_fn,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_ground_truths,
                group_size=args.group_size,
                advantage_eps=args.advantage_eps,
                normalize_by_std=args.normalize_by_std,
                processes=args.num_proc,
            )

            if args.loss_type == "no_baseline":
                raw_rewards = raw_rewards.to(args.policy_device)
                advantages = None
            else:
                raw_rewards = None
                advantages = advantages.to(args.policy_device)
            
            if args.loss_type == "grpo_clip":
                cliprange = args.cliprange
                old_log_probs = get_rollout_logprobs(ref_outputs
                                                     ).to(args.policy_device)
            else:
                cliprange = None
                old_log_probs = None
       
            policy_log_probs = get_policy_log_probs(policy, 
                                                    ref_outputs,
                                                    tokenizer.pad_token_id,
                                                    micro_train_batch_size)

            loss, metadata = grpo_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=prepare_mask(ref_outputs,args.policy_device),
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                loss_type=args.loss_type,
                raw_rewards=raw_rewards,
                advantages=advantages,
                old_log_probs=old_log_probs,
                cliprange=cliprange,
            )
            metadata.update(reward_metadata)
            mean_metadata += metadata
            counter += 1
            progress_bar.set_postfix({"loss": loss.item()})
            progress_bar.update(1)
            
            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), max_norm=args.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()

            if step % args.metadata_wandb_log_step == 0:
                clear_device_cache(garbage_collection=True)
                mean_metadata = mean_metadata / counter
                tqdm.write(f"Step {step} metadata: {mean_metadata}")
                wandb.log(
                    {f"train/{k}": v for k, v in mean_metadata.items()}, step=step
                )
                mean_metadata = SummableDict()
                counter = 0

            if step % args.logging_step == 0:
                clear_device_cache(garbage_collection=True)
                log_results = log_generations(
                    model=ref_model,
                    prompts=samples["prompt"],
                    ground_truths=samples["response"],
                    parallel_core=parallel_log_core,
                    sampling_params=eval_sampling_params,
                    num_log=args.num_log,
                )
                tqdm.write(f"Logging generations at step {step}:")
                tqdm.write(f"Stats: {log_results['stats']}")
                tqdm.write("Samples:")
                for sample in log_results["samples"]:
                    tqdm.write(f"Prompt: {sample['prompt']}")
                    tqdm.write(f"Response: {sample['response']}")
                    tqdm.write(f"Ground Truth: {sample['ground_truth']}")
                    tqdm.write(f"Reward: {sample['reward']}")
                    tqdm.write(f"Format Reward: {sample['format_reward']}")
                    tqdm.write(f"Answer Reward: {sample['answer_reward']}")
                    tqdm.write(f"Sample Entropy: {sample['sample_entropy']}")
                    tqdm.write(f"Response Length: {sample['response_len']}")
                    tqdm.write("-----")
            

def main():
    argparser = ArgumentParser(description="GRPO Training")
    argparser.add_argument(
        "--policy_model_id",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="HuggingFace model ID for the policy being trained",
    )
    argparser.add_argument(
        "--ref_model_id",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="HuggingFace model ID for the reference model used for scoring and generation during evaluation/logging",
    )
    argparser.add_argument(
        "--tokenizer_id",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="HuggingFace model ID for the tokenizer (often the same as policy_model_id)",
    )
    argparser.add_argument(
        "--dataset_name",
        type=str,
        default="hkust-nlp/dart-math-uniform",
        help="HuggingFace dataset name (e.g. 'Dahoas/rm-static')",
    )
    argparser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use for training (e.g. 'train', 'test')",
    )
    argparser.add_argument(
        "--policy_device",
        type=str,
        default="cuda:0",
        help="Device for the policy model (e.g. 'cuda:0')",
    )
    argparser.add_argument(
        "--vllm_device",
        type=str,
        default="cuda:1",
        help="Device for the vLLM reference model (e.g. 'cuda:1')",
    )
    argparser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for the optimizer",
    )
    argparser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        help="Optimizer to use (e.g. 'adamw', 'sgd')",
    )
    argparser.add_argument(
        "--train_batch_size", type=int, default=256, help="Batch size for training"
    )
    argparser.add_argument(
        "--eval_batch_size", type=int, default=512, help="Batch size for evaluation"
    )
    argparser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    argparser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=128,
        help="Number of microbatches to accumulate before each optimizer step",
    )
    argparser.add_argument(
        "--group_size",
        type=int,
        default=8,
        help="Number of responses per question (group) for reward normalization",
    )
    argparser.add_argument(
        "--loss_type",
        type=str,
        default="grpo_clip",
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        help="Type of policy gradient loss to use",
    )
    argparser.add_argument(
        "--cliprange",
        type=float,
        default=0.2,
        help="Clip parameter ε for GRPO-Clip loss (ignored for other loss types)",
    )
    argparser.add_argument(
        "--sampling_temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for generation during training and evaluation",
    )
    argparser.add_argument(
        "--sampling_max_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate during training and evaluation",
    )
    argparser.add_argument(
        "--sampling_min_tokens",
        type=int,
        default=4,
        help="Minimum number of tokens to generate during training and evaluation",
    )
    argparser.add_argument(
        "--normalize_by_std",
        action="store_true",
        default=True,
        help="Whether to normalize rewards by the per-group standard deviation (in addition to subtracting the mean)",
    )
    argparser.add_argument("--early_stopping", action="store_true", default=True)
    argparser.add_argument("--early_stopping_metric", type=str, default="reward")
    argparser.add_argument("--early_stopping_patience", type=int, default=3)
    argparser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    argparser.add_argument(
        "--advantage_eps",
        type=float,
        default=1e-6,
        help="Small constant to avoid division by zero in normalization."
    )
    argparser.add_argument(
        "--metadata_wandb_log_step",
        type=int,
        default=10000,
        help="Number of steps between logging metadata to Weights & Biases",
    )
    argparser.add_argument(
        "--eval_step",
        type=int,
        default=100000,
        help="Number of steps between evaluations",
    )
    argparser.add_argument(
        "--logging_step",
        type=int,
        default=50000,
        help="Number of steps between logging generations",
    )
    argparser.add_argument(
        "--ref_sync_steps",
        type=int,
        default=5000,
        help="Number of steps between synchronizing the reference model with the policy",
    )
    argparser.add_argument(
        "--num_log",
        type=int,
        default=8,
        help="Number of samples to log during generation logging",
    )
    argparser.add_argument(
        "--output_dir",
        type=str,
        default="./sft_checkpoints",
        help="Directory to save checkpoints",
    )
    argparser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    argparser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.85,
        help="GPU memory utilization for vLLM (between 0 and 1)",
    )
    argparser.add_argument(
        "--normalize_constant",
        type=float,
        default=1.0,
        help="Constant for normalizing rewards",
    )
    argparser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Test size for train/test split if the dataset does not have a predefined test split",
    )
    argparser.add_argument(
        "--num_proc",
        type=int,
        default=24,
        help="Number of processes to use for dataset mapping",
    )
    argparser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )
    args = argparser.parse_args()
    wandb.login()
    wandb.init(project="cs336_grpo", config=vars(args))
    grpo_training(args)


if __name__ == "__main__":
    main()
