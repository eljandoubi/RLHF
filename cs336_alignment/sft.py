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
