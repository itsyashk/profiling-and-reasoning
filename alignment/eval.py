from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .prompts import COT_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE
from .rewards import answer_tag_reward_fn, majority_vote_tagged_answers


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_VALIDATION_SIZE = 256


def load_gsm8k_examples(split: str) -> list[dict[str, Any]]:
    """Load GSM8K examples from HuggingFace datasets."""
    from datasets import load_dataset
    return list(load_dataset("openai/gsm8k", "main", split=split))


def build_prompts(examples: Sequence[dict[str, Any]], prompt_template: str) -> list[str]:
    """Format raw GSM8K examples into prompt strings."""
    return [str(prompt_template).format(question=ex["question"]) for ex in examples]


def _extract_ground_truth(examples: Sequence[dict[str, Any]]) -> list[str]:
    truths = []
    for ex in examples:
        answer = ex["answer"]
        if "####" in answer:
            answer = answer.split("####")[-1].strip()
        truths.append(answer)
    return truths


# ---------------------------------------------------------------------------
# vLLM path (fast batched inference, Linux/Colab only)
# ---------------------------------------------------------------------------

def evaluate_vllm(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: Sequence[str],
    eval_sampling_params,
    ground_truths: Sequence[str] | None = None,
    num_return_sequences: int = 1,
) -> dict[str, Any]:
    """Generate with vLLM, score responses, return evaluation artifacts."""
    outputs = vllm_model.generate(list(prompts), eval_sampling_params)

    results = []
    total_format = 0.0
    total_answer = 0.0

    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        gt = ground_truths[i] if ground_truths is not None else ""

        if num_return_sequences == 1:
            response = output.outputs[0].text
            reward_info = reward_fn(response, gt)
            results.append({"prompt": prompt, "response": response, "ground_truth": gt, **reward_info})
            total_format += reward_info["format_reward"]
            total_answer += reward_info["answer_reward"]
        else:
            responses = [o.text for o in output.outputs]
            voted = majority_vote_tagged_answers(responses)
            voted_response = f"<answer> {voted} </answer>" if voted else ""
            reward_info = reward_fn(voted_response, gt)
            results.append({
                "prompt": prompt,
                "responses": responses,
                "voted_answer": voted,
                "ground_truth": gt,
                **reward_info,
            })
            total_format += reward_info["format_reward"]
            total_answer += reward_info["answer_reward"]

    n = len(prompts)
    return {
        "results": results,
        "mean_format_reward": total_format / n,
        "mean_answer_reward": total_answer / n,
        "n": n,
    }


def _load_vllm_model(model_name: str = DEFAULT_MODEL_NAME):
    from vllm import LLM
    return LLM(model=model_name, dtype="bfloat16")


def _make_sampling_params(max_tokens: int, temperature: float = 1.0, n: int = 1):
    from vllm import SamplingParams
    return SamplingParams(
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )


# ---------------------------------------------------------------------------
# Transformers fallback path (slower, CPU/any GPU)
# ---------------------------------------------------------------------------

def _generate_with_transformers(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    num_return_sequences: int = 1,
    device: str = "cuda",
) -> list[list[str]]:
    import torch
    stop = "</answer>"
    model.eval()
    all_responses: list[list[str]] = []

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]
            do_sample = num_return_sequences > 1 or temperature != 1.0
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            responses = []
            for seq in outputs:
                text = tokenizer.decode(seq[input_len:], skip_special_tokens=False)
                if stop in text:
                    text = text[: text.index(stop) + len(stop)]
                responses.append(text)
            all_responses.append(responses)

    return all_responses


def evaluate_transformers(
    model,
    tokenizer,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    max_new_tokens: int = 1024,
    num_return_sequences: int = 1,
    temperature: float = 1.0,
    device: str = "cuda",
) -> dict[str, Any]:
    all_responses = _generate_with_transformers(
        model, tokenizer, prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
        device=device,
    )
    results = []
    total_format = total_answer = 0.0

    for prompt, responses, gt in zip(prompts, all_responses, ground_truths):
        if num_return_sequences == 1:
            response = responses[0]
            reward_info = reward_fn(response, gt)
            results.append({"prompt": prompt, "response": response, "ground_truth": gt, **reward_info})
        else:
            voted = majority_vote_tagged_answers(responses)
            voted_response = f"<answer> {voted} </answer>" if voted else ""
            reward_info = reward_fn(voted_response, gt)
            results.append({
                "prompt": prompt, "responses": responses,
                "voted_answer": voted, "ground_truth": gt, **reward_info,
            })
        total_format += reward_info["format_reward"]
        total_answer += reward_info["answer_reward"]

    n = len(prompts)
    return {
        "results": results,
        "mean_format_reward": total_format / n,
        "mean_answer_reward": total_answer / n,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def write_evaluation_results(results: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved → {output_path}")


def _load_model_and_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Named runners (used by run_*.py scripts or notebook)
# ---------------------------------------------------------------------------

def run_direct_baseline(output_path: Path) -> None:
    examples = load_gsm8k_examples("test")
    prompts = build_prompts(examples, DIRECT_PROMPT_TEMPLATE)
    ground_truths = _extract_ground_truth(examples)
    model, tokenizer = _load_model_and_tokenizer()
    device = next(model.parameters()).device.type
    results = evaluate_transformers(
        model, tokenizer, answer_tag_reward_fn, prompts, ground_truths,
        max_new_tokens=512, device=device,
    )
    print(f"format: {results['mean_format_reward']:.3f}  answer: {results['mean_answer_reward']:.3f}")
    write_evaluation_results(results, output_path)


def run_cot_baseline(output_path: Path) -> None:
    examples = load_gsm8k_examples("test")
    prompts = build_prompts(examples, COT_PROMPT_TEMPLATE)
    ground_truths = _extract_ground_truth(examples)
    model, tokenizer = _load_model_and_tokenizer()
    device = next(model.parameters()).device.type
    results = evaluate_transformers(
        model, tokenizer, answer_tag_reward_fn, prompts, ground_truths,
        max_new_tokens=1024, device=device,
    )
    print(f"format: {results['mean_format_reward']:.3f}  answer: {results['mean_answer_reward']:.3f}")
    write_evaluation_results(results, output_path)


def run_self_consistency_baseline(output_path: Path, k: int = 5) -> None:
    examples = load_gsm8k_examples("test")
    prompts = build_prompts(examples, COT_PROMPT_TEMPLATE)
    ground_truths = _extract_ground_truth(examples)
    model, tokenizer = _load_model_and_tokenizer()
    device = next(model.parameters()).device.type
    results = evaluate_transformers(
        model, tokenizer, answer_tag_reward_fn, prompts, ground_truths,
        max_new_tokens=1024, num_return_sequences=k, temperature=1.0, device=device,
    )
    results["k"] = k
    print(f"format: {results['mean_format_reward']:.3f}  answer: {results['mean_answer_reward']:.3f}")
    write_evaluation_results(results, output_path)


def get_prompt_template(use_cot: bool) -> str:
    return COT_PROMPT_TEMPLATE if use_cot else DIRECT_PROMPT_TEMPLATE
