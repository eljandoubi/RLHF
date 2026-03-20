from argparse import Namespace
from typing import Any, Callable
from unittest.mock import patch

import torch
import torch.optim as optim
from datasets import load_dataset
from drgrpo_grader import r1_zero_reward_fn
from evaluation import evaluate_vllm, summarize_results
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed


def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer)-> dict[str, torch.Tensor]:
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
    
    tokenized_prompts = tokenizer(prompt_strs,return_attention_mask=False)
    tokenized_outputs = tokenizer(output_strs,return_attention_mask=False)
    tokenized_inputs = [prompt + output for prompt, output in zip(tokenized_prompts["input_ids"], tokenized_outputs["input_ids"])]
    mask = [[0]*len(prompt) + [1]*len(output) for prompt, output in zip(tokenized_prompts["input_ids"], tokenized_outputs["input_ids"])]
    del tokenized_prompts, tokenized_outputs
    tokenized_inputs = torch.nn.utils.rnn.pad_sequence([torch.tensor(input) for input in tokenized_inputs], batch_first=True, padding_value=tokenizer.pad_token_id)
    mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(m) for m in mask], batch_first=True, padding_value=0).long()

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
    # probs = torch.nn.functional.softmax(logits, dim=-1)
    # log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    # entropy = -torch.sum(probs * log_probs, dim=-1)
    # return entropy
    maxes, _ = logits.max(dim=-1, keepdim=True)
    shift_logits = logits - maxes
    del maxes
    normalized_logits = shift_logits - torch.logsumexp(shift_logits, dim=-1, keepdim=True)
    del shift_logits
    return -(normalized_logits.exp()*normalized_logits).sum(dim=-1)

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
    logits:torch.Tensor = model_output.logits

    maxes, _ = logits.max(dim=-1, keepdim=True)
    shift_logits = logits - maxes
    del maxes
    normalized_logits = shift_logits - torch.logsumexp(shift_logits, dim=-1, keepdim=True)
    del shift_logits
    log_probs = normalized_logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    output = {"log_probs": log_probs}
    if return_token_entropy:
        output["token_entropy"] = -(normalized_logits.exp()*normalized_logits).sum(dim=-1)
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute a forward-and-backward pass on a microbatch.
    Args:
    policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
    SFT policy being trained.
    response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
    prompt/padding.
    gradient_accumulation_steps Number of microbatches per optimizer step.
    normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0.
    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
    loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
    this so we can log it.
    metadata Dict with metadata from the underlying loss call, and any other statistics you
    might want to log."""

    loss = -masked_normalize(tensor=policy_log_probs, mask=response_mask, normalize_constant=normalize_constant*gradient_accumulation_steps, dim=1).mean() 
    loss.backward()
    return loss.detach(), {}



def log_generations(
    model: LLM,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    *,
    sampling_params: SamplingParams,
    num_log: int = 8,
    step: int | None = None,
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

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    stats = {
        "step": step,
        "avg_response_len": _mean([s["response_len"] for s in samples]),
        "avg_response_len_correct": _mean([s["response_len"] for s in samples if s["answer_reward"] >= 1.0]),
        "avg_response_len_wrong": _mean([s["response_len"] for s in samples if s["answer_reward"] < 1.0]),
        "avg_token_entropy": _mean([s["sample_entropy"] for s in samples]),
        "avg_reward": _mean([s["reward"] for s in samples]),
        "avg_format_reward": _mean([s["format_reward"] for s in samples]),
        "avg_answer_reward": _mean([s["answer_reward"] for s in samples]),
        "n_logged": len(samples),
    }

    return {"samples": samples, "stats": stats}


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
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
        return_value=None
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
    optimizer: torch.optim.Optimizer = optimizer_cls(policy.parameters(), lr=args.learning_rate)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    ref_model = init_vllm(args.ref_model_id, device=args.vllm_device, seed=args.seed)
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"],
        include_stop_str_in_output=True, logprobs=1
        )
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    dataset.rename_column("query", "prompt")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"].shuffle(seed=args.seed)
    eval_dataset = dataset["test"]
    policy.train()
    len_train_dataset = len(train_dataset)
    step_per_epoch = len_train_dataset // args.train_batch_size + int(len_train_dataset % args.train_batch_size == 0)
    total_steps = step_per_epoch * args.epochs
    progress_bar = tqdm(total=total_steps, desc="SFT Training") 
    for i in range(args.epochs):
        for j, samples in enumerate(train_dataset.iter(batch_size=args.train_batch_size)):
            tokenized = tokenize_prompt_and_output(samples["prompt"], samples["response"], tokenizer)
            input_ids = tokenized["input_ids"].to(args.policy_device)
            labels = tokenized["labels"].to(args.policy_device)
            response_mask = tokenized["response_mask"].to(args.policy_device)
            policy_outputs = get_response_log_probs(policy, input_ids, labels, return_token_entropy=False)
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_outputs["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                normalize_constant=1.0,
            )
            progress_bar.set_postfix({"loss": loss.item()})
            progress_bar.update(1)
            step = i * step_per_epoch + j + 1
            if step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step % args.eval_step == 0:
                load_policy_into_vllm_instance(policy, ref_model)
                eval_results = []
                for eval_samples in eval_dataset.iter(batch_size=args.eval_batch_size):
                    eval_results.extend(evaluate_vllm(ref_model, r1_zero_reward_fn, eval_samples, sampling_params))
                avg_scores = summarize_results(eval_results)
                tqdm.write("Average Scores:")
                for metric, score in avg_scores.items():
                    tqdm.write(f"  {metric}: {score:.4f}")
                save_path = f"{args.output_dir}/checkpoint-{step}"
                policy.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                tqdm.write(f"Saved checkpoint to {save_path}")

            