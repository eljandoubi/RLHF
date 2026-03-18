from typing import Any, Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


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


@torch.inference_mode()
def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool | None = None,
    num_log: int = 8,
    step: int | None = None,
    stop_str: str | None = None,
    device: torch.device | str | None = None,
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

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    n = min(num_log, len(prompts))
    prompts = prompts[:n]
    ground_truths = ground_truths[:n]

    # decide sampling
    if do_sample is None:
        do_sample = temperature > 0

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # batch tokenize to get attention mask
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)    

    # generate in one batch
    gen_out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    sequences = gen_out.sequences  # (B, T_total)
    prompt_lens = attention_mask.sum(dim=1).tolist()

    samples: list[dict[str, Any]] = []
    lengths: list[int] = []
    lengths_correct: list[int] = []
    lengths_wrong: list[int] = []
    entropies: list[float] = []

    # Compute per-step entropies from scores
    avg_ent_per_sample = [0.0 for _ in range(n)]
    if gen_out.scores is not None and len(gen_out.scores) > 0:
        # accumulate entropies per step per sample
        acc = [0.0 for _ in range(n)]
        for step_logits in gen_out.scores:
            for i in range(n):
                acc[i] += float(compute_entropy(step_logits[i]).item())
        denom = float(len(gen_out.scores))
        avg_ent_per_sample = [x / denom for x in acc]

    for i in range(n):
        prompt = prompts[i]
        gt = ground_truths[i]
        pl = int(prompt_lens[i])

        full_ids = sequences[i]
        gen_ids = full_ids[pl:]  # generated part

        response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        if stop_str is not None and stop_str in response_text:
            response_text = response_text.split(stop_str)[0] + stop_str

        reward_dict = reward_fn(response_text, gt)

        gen_len = int(gen_ids.numel())
        avg_ent = float(avg_ent_per_sample[i])

        samples.append(
            {
                "step": step,
                "prompt": prompt,
                "response": response_text,
                "ground_truth": gt,
                "reward": float(reward_dict.get("reward", 0.0)),
                "format_reward": float(reward_dict.get("format_reward", 0.0)),
                "answer_reward": float(reward_dict.get("answer_reward", 0.0)),
                "avg_token_entropy": avg_ent,
                "response_len": gen_len,
            }
        )

        lengths.append(gen_len)
        entropies.append(avg_ent)
        is_correct = float(reward_dict.get("answer_reward", 0.0)) >= 1.0
        if is_correct:
            lengths_correct.append(gen_len)
        else:
            lengths_wrong.append(gen_len)

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    stats = {
        "step": step,
        "avg_response_len": _mean([float(x) for x in lengths]),
        "avg_response_len_correct": _mean([float(x) for x in lengths_correct]),
        "avg_response_len_wrong": _mean([float(x) for x in lengths_wrong]),
        "avg_token_entropy": _mean(entropies),
        "avg_reward": _mean([s["reward"] for s in samples]),
        "avg_format_reward": _mean([s["format_reward"] for s in samples]),
        "avg_answer_reward": _mean([s["answer_reward"] for s in samples]),
        "n_logged": len(samples),
    }

    return {"samples": samples, "stats": stats}