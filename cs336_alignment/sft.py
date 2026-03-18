import torch
from transformers import PreTrainedTokenizer


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