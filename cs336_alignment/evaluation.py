import json
import math
from argparse import ArgumentParser, Namespace
from itertools import batched
from typing import Any, Callable, Dict, List

from tqdm import tqdm
from vllm import LLM, SamplingParams

from .summable_dict import dict_mean

BATCH_SIZE = 8


def r1_format_response(response: str) -> str:
    if response.count("\n#### ") < 1:
        return f"</think> <answer>{response}</answer>"
    thinking, answer = response.split("\n#### ")
    return f"{thinking}</think> <answer>{answer}</answer>"


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    samples: List[Dict[str, str]],
    eval_sampling_params: SamplingParams,
) -> List[Dict[str, Any]]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    with open("cs336_alignment/prompts/r1_zero.prompt") as f:
        prompt_template = f.read()
    results: List[Dict[str, Any]] = []
    total = int(math.ceil(len(samples) * 1.0 / BATCH_SIZE))
    for batch_samples in tqdm(
        batched(samples, BATCH_SIZE), total=total, desc="Batch evaluate the data."
    ):
        prompts = [
            prompt_template.format(question=sample["question"])
            for sample in batch_samples
        ]
        ground_truths = [
            r1_format_response(sample["answer"]) for sample in batch_samples
        ]
        responses = vllm_model.generate(prompts, eval_sampling_params, use_tqdm=False)
        for i, response in enumerate(responses):
            prompt = response.prompt
            answer = response.outputs[0].text
            scores = reward_fn(answer, ground_truths[i])
            results.append(
                {
                    "exemple": prompt,
                    "ground_truth": ground_truths[i],
                    "generation": answer,
                    "scores": scores,
                }
            )

    return results


def main(args: Namespace):
    print(args)
    from drgrpo_grader import r1_zero_reward_fn

    llm = LLM(args.model_name)

    with open(args.dataset) as f:
        raws = f.readlines()

    dataset = [json.loads(raw) for raw in raws]

    del raws

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    results = evaluate_vllm(llm, r1_zero_reward_fn, dataset, sampling_params)

    with open("math_baseline_eval.json", "w") as f:
        json.dump(results, f)

    avg_scores = dict_mean(results)
    print("Average Scores:")
    for metric, score in avg_scores["scores"].items():
        print(f"  {metric}: {score:.4f}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluation of the baseline on math.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--dataset", type=str, default="data/gsm8k/test.jsonl")
    args = parser.parse_args()
    main(args)
