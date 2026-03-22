import argparse
from argparse import Namespace
from typing import Any, Callable, Literal
from unittest.mock import patch

import torch
import torch.optim as optim
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

import wandb
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.summable_dict import SummableDict, dict_mean
from cs336_alignment.utils import format_sample


def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for other tokens (prompt or padding).
    Args:
    prompt_strs: list[str] List of prompt strings.
    output_strs: list[str] List of output strings.
    tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.
    Returns:
    dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of
    the tokenized prompt and output strings. Then the returned dictionary should have the
    following keys:
    input_ids torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
    the tokenized prompt and output strings, with the final token sliced off.
    labels torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
    shifted input ids, i.e., the input ids without the first token.
    response_mask torch.Tensor of shape (batch_size, max(prompt_and_output_lens) -
    1): a mask on the response tokens in the labels."""

    tokenized_prompts = tokenizer(prompt_strs, return_attention_mask=False)
    tokenized_outputs = tokenizer(output_strs, return_attention_mask=False)
    tokenized_inputs = [
        prompt + output
        for prompt, output in zip(
            tokenized_prompts["input_ids"], tokenized_outputs["input_ids"]
        )
    ]
    mask = [
        [0] * len(prompt) + [1] * len(output)
        for prompt, output in zip(
            tokenized_prompts["input_ids"], tokenized_outputs["input_ids"]
        )
    ]
    del tokenized_prompts, tokenized_outputs
    tokenized_inputs = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(input) for input in tokenized_inputs],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    mask = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(m) for m in mask], batch_first=True, padding_value=0
    ).long()

    return {
        "input_ids": tokenized_inputs[:, :-1],
        "labels": tokenized_inputs[:, 1:].clone(),
        "response_mask": mask[:, 1:],
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    Args:
    logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
    containing unnormalized logits.
    Returns:
    torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
    prediction."""

    maxes, _ = logits.max(dim=-1, keepdim=True)
    shift_logits = logits - maxes
    del maxes
    normalized_logits = shift_logits - torch.logsumexp(
        shift_logits, dim=-1, keepdim=True
    )
    del shift_logits
    return -(normalized_logits.exp() * normalized_logits).sum(dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Args:
    model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device
    and in inference mode if gradients should not be computed).
    input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt +
    response tokens as produced by your tokenization method.
    labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your
    tokenization method.
    return_token_entropy: bool If True, also return per-token entropy by calling
    compute_entropy.
    Returns:
    dict[str, torch.Tensor].
    "log_probs" shape (batch_size, sequence_length), conditional log-probabilities
    log pθ(xt |x<t).
    "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy
    for each position (present only if return_token_entropy=True)"""

    model_output = model(input_ids=input_ids)
    logits: torch.Tensor = model_output.logits

    maxes, _ = logits.max(dim=-1, keepdim=True)
    shift_logits = logits - maxes
    del maxes
    normalized_logits = shift_logits - torch.logsumexp(
        shift_logits, dim=-1, keepdim=True
    )
    del shift_logits
    log_probs = normalized_logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    output = {"log_probs": log_probs}
    if return_token_entropy:
        output["token_entropy"] = -(normalized_logits.exp() * normalized_logits).sum(
            dim=-1
        )
    return output


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant, considering only those elements where mask
    == 1.
    Args:
    tensor: torch.Tensor The tensor to sum and normalize.
    mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
    normalize_constant: float the constant to divide by for normalization.
    dim: int | None the dimension to sum along before normalization. If None, sum over all
    dimensions.
    Returns:
    torch.Tensor the normalized sum, where masked elements (mask == 0) don’t contribute to
    the sum."""
    return (tensor * mask.to(tensor.dtype)).sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Execute a forward-and-backward pass on a microbatch.
    Args:
    policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
    SFT policy being trained.
    response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
    prompt/padding.
    gradient_accumulation_steps Number of microbatches per optimizer step.
    normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0.
    Returns:
    tuple[torch.Tensor, dict[str, float]].
    loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
    this so we can log it.
    metadata Dict with metadata from the underlying loss call, and any other statistics you
    might want to log."""

    loss = -masked_normalize(
        tensor=policy_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant * gradient_accumulation_steps,
        dim=1,
    ).mean()
    loss.backward()
    with torch.inference_mode():
        loss_item = loss.item()
        resp_tokens = response_mask.to(policy_log_probs.dtype).sum().item()
        metadata = {
            "raw_nll": loss_item * normalize_constant * gradient_accumulation_steps,
            "scaled_nll": loss_item * gradient_accumulation_steps,
            "loss": loss_item,
            "resp_tokens": resp_tokens,
            "avg_log_prob": loss_item
            * normalize_constant
            * gradient_accumulation_steps
            / resp_tokens
            if resp_tokens > 0
            else 0.0,
        }
    return loss.detach(), metadata


def log_generations(
    model: LLM,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    *,
    sampling_params: SamplingParams,
    num_log: int | None = None,
    step: int | None = None,
    return_obejcts: Literal["samples", "stats", "both"] = "both",
) -> dict[str, Any]:
    """
    Generate responses for a few prompts and log:
      - prompt / response / ground_truth
      - reward: format_reward, answer_reward, reward
      - avg token entropy over generated tokens
      - length stats (avg, correct avg, wrong avg)

    Returns a dict with:
      - "samples": list[dict]
      - "stats": dict
    """
    assert len(prompts) == len(ground_truths), "prompts and ground_truths must align"

    if num_log is not None:
        n = min(num_log, len(prompts))
        prompts = prompts[:n]
        ground_truths = ground_truths[:n]

    # generate responses with vLLM
    responses = model.generate(prompts, sampling_params, use_tqdm=False)

    samples: list[dict[str, Any]] = []
    stop_str = sampling_params.stop[0] if sampling_params.stop else None
    for i, response in enumerate(responses):
        prompt = prompts[i]
        gt = ground_truths[i]
        response_text = response.outputs[0].text

        if stop_str is not None and stop_str in response_text:
            response_text = response_text.split(stop_str)[0] + stop_str

        reward_dict = reward_fn(response_text, gt)

        if tokenizer is not None:
            response_len = len(tokenizer(response_text).input_ids)
        else:
            response_len = len(response_text.split())

        logprobs = response.outputs[0].logprobs
        if logprobs is not None:
            smp_ent = -sum(lp.logprob for dt in logprobs for lp in dt.values())
        else:
            smp_ent = 0.0

        samples.append(
            {
                "step": step,
                "prompt": prompt,
                "response": response_text,
                "ground_truth": gt,
                "reward": reward_dict.get("reward", 0.0),
                "format_reward": reward_dict.get("format_reward", 0.0),
                "answer_reward": reward_dict.get("answer_reward", 0.0),
                "sample_entropy": smp_ent,
                "response_len": response_len,
            }
        )
    if return_obejcts == "both":
        return {"samples": samples, "stats": dict_mean(samples)}
    if return_obejcts == "samples":
        return {"samples": samples}
    if return_obejcts == "stats":
        return {"stats": dict_mean(samples)}


def init_vllm(
    model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85
):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    13
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def get_optimizer(optimizer_name: str) -> torch.optim.Optimizer:
    """
    Return an optimizer class based on the optimizer name.
    """
    if optimizer_name == "adamw":
        return optim.AdamW
    elif optimizer_name == "sgd":
        return optim.SGD
    elif optimizer_name == "adagrad":
        return optim.Adagrad
    elif optimizer_name == "rmsprop":
        return optim.RMSprop
    elif optimizer_name == "adam":
        return optim.Adam
    elif optimizer_name == "adamax":
        return optim.Adamax
    elif optimizer_name == "nadam":
        return optim.NAdam
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def prepare_data(
    source: str, test_size: float = 0.2, seed: int = 42, num_proc: int = 4
) -> dict[str, Any]:
    dataset = load_dataset(source)
    if len(dataset) == 1:
        assert "train" in dataset, (
            "Expected dataset to have a 'train' split if it has only one split"
        )
        dataset = dataset["train"]
        dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    else:
        assert "train" in dataset and "test" in dataset, (
            "Expected dataset with multiple splits to have 'train' and 'test' splits"
        )
    if "query" in dataset["test"].column_names:
        dataset = dataset.rename_column("query", "prompt")
    if "answer" in dataset["test"].column_names:
        dataset = dataset.rename_column("answer", "response")
    if "question" in dataset["test"].column_names:
        dataset = dataset.rename_column("question", "prompt")
    if "solution" in dataset["test"].column_names:
        dataset = dataset.rename_column("solution", "response")
    dataset["train"].shuffle(seed=seed)
    dataset = dataset.map(format_sample, num_proc=num_proc)
    return dataset


def sft_training(args: Namespace):
    """
    Main SFT training loop. You should call the above functions from here to implement the training loop.
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
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
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
    train_step_per_epoch = len_train_dataset // args.train_batch_size + int(
        len_train_dataset % args.train_batch_size > 0
    )
    len_eval_dataset = len(eval_dataset)
    eval_step_per_epoch = len_eval_dataset // args.eval_batch_size + int(
        len_eval_dataset % args.eval_batch_size > 0
    )
    total_steps = train_step_per_epoch * args.epochs
    progress_bar = tqdm(total=total_steps, desc="SFT Training")
    mean_metadata = SummableDict()
    counter = 0
    for i in range(args.epochs):
        for j, samples in enumerate(
            train_dataset.iter(batch_size=args.train_batch_size)
        ):
            tokenized = tokenize_prompt_and_output(
                samples["prompt"], samples["response"], tokenizer
            )
            input_ids = tokenized["input_ids"].to(args.policy_device)
            labels = tokenized["labels"].to(args.policy_device)
            response_mask = tokenized["response_mask"].to(args.policy_device)
            policy_outputs = get_response_log_probs(
                policy, input_ids, labels, return_token_entropy=False
            )
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_outputs["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                normalize_constant=args.normalize_constant,
            )
            mean_metadata += metadata
            counter += 1
            progress_bar.set_postfix({"loss": loss.item()})
            progress_bar.update(1)
            step = i * train_step_per_epoch + j + 1
            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), max_norm=args.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()

            if step % args.metadata_wandb_log_step == 0:
                mean_metadata = mean_metadata / counter
                tqdm.write(f"Step {step} metadata: {mean_metadata}")
                wandb.log(
                    {f"train/{k}": v for k, v in mean_metadata.items()}, step=step
                )
                mean_metadata = SummableDict()
                counter = 0

            if step % args.eval_step == 0:
                tqdm.write(f"Evaluating at step {step}...")
                load_policy_into_vllm_instance(policy, ref_model)
                eval_results = []
                for eval_samples in tqdm(eval_dataset.iter(batch_size=args.eval_batch_size), desc="Evaluating",
                                         total=eval_step_per_epoch):
                    eval_results.extend(
                        log_generations(
                            model=ref_model,
                            tokenizer=tokenizer,
                            prompts=eval_samples["prompt"],
                            ground_truths=eval_samples["response"],
                            reward_fn=r1_zero_reward_fn,
                            sampling_params=sampling_params,
                            step=step,
                            return_obejcts="samples",
                        )["samples"]
                    )
                avg_scores = dict_mean(eval_results)
                del eval_results
                tqdm.write("Average Scores:")
                for metric, score in avg_scores.items():
                    tqdm.write(f"  {metric}: {score:.4f}")
                wandb.log({f"eval/{k}": v for k, v in avg_scores.items()}, step=step)
                save_path = f"{args.output_dir}/checkpoint-{step}"
                policy.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                tqdm.write(f"Saved checkpoint to {save_path}")

            if step % args.logging_step == 0:
                load_policy_into_vllm_instance(policy, ref_model)
                log_results = log_generations(
                    model=ref_model,
                    tokenizer=tokenizer,
                    prompts=samples["prompt"],
                    ground_truths=samples["response"],
                    reward_fn=r1_zero_reward_fn,
                    sampling_params=sampling_params,
                    num_log=args.num_log,
                    step=step,
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
    argparser = argparse.ArgumentParser(description="SFT Training")
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
        "--train_batch_size", type=int, default=4, help="Batch size for training"
    )
    argparser.add_argument(
        "--eval_batch_size", type=int, default=64, help="Batch size for evaluation"
    )
    argparser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    argparser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of microbatches to accumulate before each optimizer step",
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
        default=1000000,
        help="Number of steps between evaluations",
    )
    argparser.add_argument(
        "--logging_step",
        type=int,
        default=50000,
        help="Number of steps between logging generations",
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
        default=0.2,
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
    wandb.init(project="cs336_sft", config=vars(args))
    sft_training(args)


if __name__ == "__main__":
    main()
