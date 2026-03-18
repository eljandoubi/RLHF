import json
import math
from argparse import ArgumentParser, Namespace
from itertools import batched
from typing import Callable, Dict, List

from tqdm import tqdm
from vllm import LLM, SamplingParams

BATCH_SIZE = 8

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    samples: List[Dict[str,str]],
    eval_sampling_params: SamplingParams
    ) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """ 
    with open("cs336_alignment/prompts/r1_zero.prompt") as f:
        prompt_template = f.read()
    results = []
    total = int(math.ceil(len(samples)*1.0/BATCH_SIZE))
    for batch_samples in tqdm(batched(samples,BATCH_SIZE),total=total,desc="Batch evaluate the data."):
        prompts = [prompt_template.format(question=sample["question"]) for sample in batch_samples]
        responses = vllm_model.generate(prompts, eval_sampling_params,use_tqdm=False)
        for response in responses:
            prompt = response.prompt
            answer = response.outputs[0].text
            scores = reward_fn(prompt,answer)
            results.append({"exemple":prompt,"generation":answer,"scores":scores})
    
    with open('math_baseline_eval.json',"w") as f:
        json.dump(results,f)

def main(args: Namespace):
    print(args)
    from drgrpo_grader import r1_zero_reward_fn

    llm = LLM(args.model_name)
    
    with open(args.dataset) as f:
        raws = f.readlines()
    
    dataset = [json.loads(raw) for raw in raws]

    del raws

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"],
        include_stop_str_in_output=True
        )
    
    evaluate_vllm(llm,r1_zero_reward_fn,dataset,sampling_params)

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluation of the baseline on math.")
    parser.add_argument("--model_name",type=str,default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--dataset",type=str,default="data/gsm8k/test.jsonl")
    args = parser.parse_args()
    main(args)