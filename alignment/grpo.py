from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, Tensor]:
    """Tokenize prompt/output pairs and build a response mask over the labels."""
    pairs = []
    for prompt, output in zip(prompt_strs, output_strs, strict=True):
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        o_ids = tokenizer.encode(output, add_special_tokens=False)
        pairs.append((p_ids, o_ids))

    max_len = max(len(p) + len(o) - 1 for p, o in pairs)

    input_ids_list, labels_list, mask_list = [], [], []
    for p_ids, o_ids in pairs:
        full = p_ids + o_ids
        pad_len = max_len - (len(full) - 1)
        pad = [tokenizer.pad_token_id] * pad_len
        input_ids_list.append(full[:-1] + pad)
        labels_list.append(full[1:] + pad)
        # response_mask is True where the label is a response token
        mask_list.append([False] * (len(p_ids) - 1) + [True] * len(o_ids) + [False] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "response_mask": torch.tensor(mask_list, dtype=torch.bool),
    }


def compute_entropy(logits: Tensor) -> Tensor:
    """Compute per-token entropies over the vocabulary dimension."""
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(torch.exp(log_probs) * log_probs).sum(dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Score conditional log-probabilities for a batch of prompt/response examples."""
    logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    result: dict[str, Tensor] = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> Tensor:
    """Sum over masked elements and normalize by the provided constant."""
    return (tensor * mask).sum(dim=dim) / normalize_constant


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Compute raw rewards and per-group normalized advantages for GRPO."""
    raw_scores = [
        reward_fn(resp, gt)["reward"]
        for resp, gt in zip(rollout_responses, repeated_ground_truths, strict=True)
    ]
    raw_rewards = torch.tensor(raw_scores, dtype=torch.float32)

    grouped = raw_rewards.view(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if normalize_by_std:
        advantages = centered / (grouped.std(dim=1, keepdim=True, unbiased=False) + advantage_eps)
    else:
        advantages = centered

    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item(),
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
    }
    return advantages.reshape(-1), raw_rewards, metadata


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the per-token GRPO-Clip loss."""
    ratios = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratios = torch.clamp(ratios, 1 - cliprange, 1 + cliprange)
    adv = advantages.expand_as(policy_log_probs)
    loss = -torch.minimum(ratios * adv, clipped_ratios * adv)
    metadata = {"clip_fraction": (ratios != clipped_ratios).float().mean()}
    return loss, metadata


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Backpropagate a single GRPO microbatch loss."""
    per_token_loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    mask_f = response_mask.to(per_token_loss.dtype)
    per_example = (per_token_loss * mask_f).sum(dim=1) / response_mask.sum(dim=1)
    loss = per_example.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, metadata


def log_generations(
    prompts: Sequence[str],
    responses: Sequence[str],
    ground_truths: Sequence[str],
    reward_infos: Sequence[dict[str, float]],
    token_entropies: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Create serializable generation logs for debugging training runs."""
    logs = []
    for i, (prompt, response, gt, reward_info) in enumerate(
        zip(prompts, responses, ground_truths, reward_infos, strict=True)
    ):
        entry: dict[str, Any] = {
            "prompt": prompt,
            "response": response,
            "ground_truth": gt,
            "reward_info": reward_info,
        }
        if token_entropies is not None:
            entry["avg_token_entropy"] = token_entropies[i]
        logs.append(entry)
    return logs


def train_grpo(
    policy_model,
    tokenizer,
    train_examples: list[dict],
    val_examples: list[dict],
    reward_fn: Callable,
    prompt_template: str,
    n_grpo_steps: int = 8,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 32,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 256,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 32,
    gradient_accumulation_steps: int = 16,
    cliprange: float = 1.0,
    normalize_by_std: bool = True,
    log_every: int = 1,
    val_every: int = 5,
    val_size: int = 256,
    device: str = "cuda",
    wandb_run=None,
) -> dict[str, Any]:
    """Run the full GRPO training loop from Section 3.5."""
    import random
    from transformers import GenerationConfig

    policy_model.train()
    optimizer = torch.optim.Adam(
        policy_model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    n_prompts_per_rollout = rollout_batch_size // group_size
    micro_batch_size = train_batch_size // gradient_accumulation_steps
    n_microbatches = rollout_batch_size // micro_batch_size

    from .eval import build_prompts

    val_prompts = build_prompts(val_examples[:val_size], prompt_template)
    val_ground_truths = [ex["answer"] for ex in val_examples[:val_size]]

    history: dict[str, list] = {"step": [], "val_reward": [], "train_reward": []}

    stop_strings = ["</answer>"]

    gen_config = GenerationConfig(
        do_sample=True,
        temperature=sampling_temperature,
        max_new_tokens=sampling_max_tokens,
        min_new_tokens=sampling_min_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    def generate_responses(prompts: list[str]) -> list[str]:
        policy_model.eval()
        responses = []
        with torch.no_grad():
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                output_ids = policy_model.generate(**inputs, generation_config=gen_config)
                new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
                text = tokenizer.decode(new_ids, skip_special_tokens=False)
                # trim at stop string
                for stop in stop_strings:
                    if stop in text:
                        text = text[: text.index(stop) + len(stop)]
                responses.append(text)
        policy_model.train()
        return responses

    def evaluate(prompts: list[str], ground_truths: list[str]) -> float:
        responses = generate_responses(prompts)
        rewards = [reward_fn(r, gt)["reward"] for r, gt in zip(responses, ground_truths)]
        return sum(rewards) / len(rewards)

    for step in range(1, n_grpo_steps + 1):
        # Sample a batch of questions
        batch_examples = random.sample(train_examples, n_prompts_per_rollout)
        batch_prompts = build_prompts(batch_examples, prompt_template)
        batch_ground_truths = [ex["answer"] for ex in batch_examples]

        # Repeat each prompt group_size times
        repeated_prompts = [p for p in batch_prompts for _ in range(group_size)]
        repeated_gts = [gt for gt in batch_ground_truths for _ in range(group_size)]

        # Generate rollouts
        rollout_responses = generate_responses(repeated_prompts)

        # Compute advantages
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=normalize_by_std,
        )

        # Tokenize all rollouts
        tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].to(device)
        advantages_dev = advantages.to(device)

        # Compute old log probs (no grad)
        with torch.no_grad():
            old_out = get_response_log_probs(policy_model, input_ids, labels, return_token_entropy=False)
            old_log_probs = old_out["log_probs"].detach()

        # Training epochs on this rollout batch
        for _epoch in range(epochs_per_rollout_batch):
            optimizer.zero_grad()
            total_loss = 0.0
            for mb in range(n_microbatches):
                start = mb * micro_batch_size
                end = start + micro_batch_size
                mb_input_ids = input_ids[start:end]
                mb_labels = labels[start:end]
                mb_mask = response_mask[start:end]
                mb_adv = advantages_dev[start:end].unsqueeze(1)
                mb_old = old_log_probs[start:end]

                out = get_response_log_probs(policy_model, mb_input_ids, mb_labels)
                mb_log_probs = out["log_probs"]

                loss, _ = grpo_microbatch_train_step(
                    policy_log_probs=mb_log_probs,
                    response_mask=mb_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    advantages=mb_adv,
                    old_log_probs=mb_old,
                    cliprange=cliprange,
                )
                total_loss += loss.item()

            grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()

        train_reward = raw_rewards.mean().item()
        history["step"].append(step)
        history["train_reward"].append(train_reward)

        if step % log_every == 0:
            print(f"Step {step}: train_reward={train_reward:.3f} loss={total_loss:.4f} grad_norm={grad_norm:.3f}")

        if step % val_every == 0:
            val_reward = evaluate(val_prompts, val_ground_truths)
            history["val_reward"].append(val_reward)
            print(f"Step {step}: val_reward={val_reward:.3f}")
            if wandb_run is not None:
                wandb_run.log({"val_reward": val_reward, "train_reward": train_reward, "step": step})

    return history
