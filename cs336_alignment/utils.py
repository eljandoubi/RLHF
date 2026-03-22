import re


def clean_latex(ans: str) -> str:
    # Remove common LaTeX wrappers
    ans = ans.strip()

    # remove $...$
    ans = re.sub(r'^\$+|\$+$', '', ans)

    # remove \[ \]
    ans = re.sub(r'^\\\[|\\\]$', '', ans)

    # remove \(...\)
    ans = re.sub(r'^\\\(|\\\)$', '', ans)

    # remove trailing punctuation
    ans = ans.strip().strip('.')

    return ans.strip()


def extract_final_answer(text: str) -> str:
    # 🔥 1. STRICT: extract from <answer>
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return clean_latex(match.group(1))

    # 🔹 2. boxed answer
    boxed = re.findall(r'\\boxed\{([^}]*)\}', text)
    if boxed:
        return clean_latex(boxed[-1])

    # 🔹 3. "answer is"
    match = re.search(r'(final answer is|answer is)\s*[:\-]?\s*([^\n.]+)', text.lower())
    if match:
        return clean_latex(match.group(2))

    # 🔹 4. last equation result
    eq_match = re.findall(r'=\s*([-+]?\d+\.?\d*)', text)
    if eq_match:
        return eq_match[-1]

    # 🔹 5. fallback
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else ""

def format_r1_zero_response(response: str) -> str:
    response = response.strip()
    final_answer = extract_final_answer(response)
    return f"<think> {response} </think> <answer> {final_answer} </answer>"

with open("cs336_alignment/prompts/r1_zero.prompt") as f:
    PROMPT_TEMPLATE = f.read()


def format_sample(dict_sample: dict[str, str]) -> dict[str, str]:

    dict_sample["prompt"] = PROMPT_TEMPLATE.format(question=dict_sample["prompt"])
    dict_sample["response"] = format_r1_zero_response(dict_sample["response"])

    return dict_sample