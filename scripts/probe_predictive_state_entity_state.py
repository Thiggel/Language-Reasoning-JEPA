#!/usr/bin/env python3
"""Compare what several Stage 1 checkpoints represent, on identical positions.

Three questions, in increasing order of specificity:

1. Do the checkpoints differ at all? Cross-model linear CKA per layer, plus
   per-model geometry. Layer 12 is frozen in every Stage 1 cell, so it is a
   built-in sanity check: CKA there must be 1.
2. Do they track the evolving set of objects in the text differently? Linear
   probes for the discourse status of the current mention (given versus new),
   for which objects the prefix has already introduced, and for how long ago
   the current object was last mentioned.
3. Do they order objects differently in an embedding? k-nearest-neighbour
   entity purity, with the raw states dumped for t-SNE.

Objects are approximated without a tagger: a document's repeated content
tokens. The heuristic is crude but deterministic and applied identically to
every model, so it cannot favour one of them. Every probe is trained fresh on
the same positions with the same initialization, and a label-shuffled control
fixes the chance level that the probe itself can reach.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from textjepa.analysis.predictive_state import (
    covariance_eigenvalues,
    effective_rank,
    linear_cka,
    mean_pairwise_cosine,
)
from textjepa.data.predictive_state import load_token_blocks
from textjepa.models.action_transition import ResidualCapture
from textjepa.training.predictive_state import (
    load_stage1_checkpoint,
    require_transformers_runtime,
    teacher_forward,
)


QWEN_MODEL_ID = "Qwen/Qwen2.5-0.5B"
QWEN_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"

# Frequent function words carry discourse structure but are not objects; they
# would otherwise dominate a "repeated token" heuristic entirely.
STOPWORD_TEXT = """the a an and or but if then than that this these those of in
on at to for with from by as is are was were be been being it its his her their
he she they we you i not no nor so such only also more most other some any each
which who whom what when where while there here have has had do does did can
could would should may might must will shall about into over under after before
between during through against above below out off up down again further once
all both few many much own same too very just now"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", action="append", default=[], metavar="NAME=PATH",
        help="named Stage 1 checkpoint; repeat. 'original' is added implicitly.",
    )
    parser.add_argument("--token-blocks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-dump", type=Path)
    parser.add_argument("--layer", type=int, action="append", default=[])
    parser.add_argument("--blocks", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--minimum-mentions", type=int, default=3)
    parser.add_argument("--entities-per-block", type=int, default=8)
    parser.add_argument("--probe-steps", type=int, default=400)
    parser.add_argument("--probe-learning-rate", type=float, default=3e-3)
    parser.add_argument("--dump-positions", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    return parser.parse_args()


def object_labels(tokenizer, blocks: torch.Tensor, args) -> dict[str, torch.Tensor]:
    """Label every position by the discourse status of the token it carries.

    given_new  1 if this token repeats an object introduced earlier in the
               document, 0 if this is its first mention, -1 if not an object.
    entity     per-block object rank (0..entities_per_block-1), else -1.
    recency    tokens since the previous mention of this object, else -1.
    prefix_set multi-hot: which of the block's objects the prefix has met.
    """
    stopwords = set(STOPWORD_TEXT.split())
    blocks_count, length = blocks.shape
    given_new = torch.full((blocks_count, length), -1, dtype=torch.long)
    entity = torch.full((blocks_count, length), -1, dtype=torch.long)
    recency = torch.full((blocks_count, length), -1, dtype=torch.long)
    prefix_set = torch.zeros(
        (blocks_count, length, args.entities_per_block), dtype=torch.float32
    )
    for row in range(blocks_count):
        ids = blocks[row].tolist()
        counts: dict[int, int] = {}
        for token in ids:
            counts[token] = counts.get(token, 0) + 1
        candidates = []
        for token, count in counts.items():
            if count < args.minimum_mentions:
                continue
            text = tokenizer.decode([token]).strip().lower()
            if len(text) < 3 or not text.isalpha() or text in stopwords:
                continue
            candidates.append((count, token))
        candidates.sort(reverse=True)
        chosen = [token for _, token in candidates[:args.entities_per_block]]
        rank = {token: index for index, token in enumerate(chosen)}
        last_seen: dict[int, int] = {}
        introduced = torch.zeros(args.entities_per_block, dtype=torch.float32)
        for position, token in enumerate(ids):
            prefix_set[row, position] = introduced
            index = rank.get(token)
            if index is None:
                continue
            entity[row, position] = index
            previous = last_seen.get(token)
            given_new[row, position] = 0 if previous is None else 1
            if previous is not None:
                recency[row, position] = position - previous
            last_seen[token] = position
            introduced = introduced.clone()
            introduced[index] = 1.0
    return {
        "given_new": given_new, "entity": entity,
        "recency": recency, "prefix_set": prefix_set,
    }


@torch.no_grad()
def collect_states(model, capture, blocks, positions, layers, args) -> dict:
    """Run identical batches and keep the residual stream at each layer."""
    collected = {layer: [] for layer in layers}
    for start in range(0, len(blocks), args.batch_size):
        tokens = blocks[start:start + args.batch_size].to(args.device)
        position_ids = positions[start:start + args.batch_size].to(args.device)
        output = teacher_forward(
            model, tokens, capture=capture, attention_mask=None,
            position_ids=position_ids, use_cache=False,
        )
        for layer in layers:
            collected[layer].append(output.states[layer].float().cpu())
    return {layer: torch.cat(values) for layer, values in collected.items()}


def fit_probe(features, targets, *, task, generator, args, mask=None):
    """Train one fresh linear probe and score it on a held-out half."""
    if mask is not None:
        features, targets = features[mask], targets[mask]
    if len(features) < 64:
        return None
    order = torch.randperm(len(features), generator=generator)
    features, targets = features[order], targets[order]
    split = len(features) // 2
    # Standardize on the training half only; probes must not see held-out
    # statistics, and unnormalized residual streams are wildly heavy-tailed.
    center = features[:split].mean(0, keepdim=True)
    scale = features[:split].std(0, keepdim=True).clamp_min(1e-6)
    features = (features - center) / scale
    train_x, test_x = features[:split].to(args.device), features[split:].to(args.device)
    train_y, test_y = targets[:split].to(args.device), targets[split:].to(args.device)
    width = train_x.shape[1]
    outputs = 1 if task in ("binary", "regression") else train_y.shape[1]
    torch.manual_seed(args.seed)
    probe = nn.Linear(width, outputs).to(args.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.probe_learning_rate,
                                  weight_decay=1e-2)
    for _ in range(args.probe_steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = probe(train_x)
        if task == "binary":
            loss = nn.functional.binary_cross_entropy_with_logits(
                prediction.squeeze(-1), train_y.float()
            )
        elif task == "multilabel":
            loss = nn.functional.binary_cross_entropy_with_logits(prediction, train_y)
        else:
            loss = nn.functional.huber_loss(prediction.squeeze(-1), train_y.float())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = probe(test_x)
        if task == "binary":
            return {"auc": roc_auc(prediction.squeeze(-1), test_y),
                    "positives": float(test_y.float().mean())}
        if task == "multilabel":
            columns = [
                roc_auc(prediction[:, index], test_y[:, index].long())
                for index in range(outputs)
                if 0 < float(test_y[:, index].mean()) < 1
            ]
            return {"mean_auc": sum(columns) / max(len(columns), 1),
                    "columns": len(columns)}
        residual = prediction.squeeze(-1) - test_y.float()
        variance = test_y.float().var(unbiased=False).clamp_min(1e-9)
        return {"r2": float(1.0 - residual.var(unbiased=False) / variance)}


def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based AUC; ties receive averaged ranks."""
    labels = labels.long()
    positive = int(labels.sum())
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty(len(scores), dtype=torch.float64, device=scores.device)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64,
                                device=scores.device)
    sorted_scores = scores[order]
    start = 0
    for index in range(1, len(sorted_scores) + 1):
        if index == len(sorted_scores) or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    positive_rank = ranks[labels == 1].sum()
    return float((positive_rank - positive * (positive + 1) / 2) / (positive * negative))


def knn_entity_purity(states, entity, *, neighbours=10, sample=3000, generator=None):
    """Share of a state's nearest neighbours carrying the same object."""
    keep = entity >= 0
    states, entity = states[keep], entity[keep]
    if len(states) < neighbours + 2:
        return float("nan")
    take = min(sample, len(states))
    index = torch.randperm(len(states), generator=generator)[:take]
    states, entity = states[index], entity[index]
    states = nn.functional.normalize(states - states.mean(0, keepdim=True), dim=-1)
    similarity = states @ states.T
    similarity.fill_diagonal_(-2.0)
    top = similarity.topk(neighbours, dim=-1).indices
    return float((entity[top] == entity[:, None]).float().mean())


def main() -> None:
    args = parse_args()
    require_transformers_runtime()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    layers = sorted(set(args.layer or [12, 18, 24]))
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator().manual_seed(args.seed)
    dataset, metadata = load_token_blocks(args.token_blocks, "validation")
    chosen = torch.randperm(len(dataset), generator=generator)[:args.blocks]
    blocks = torch.stack([dataset[int(index)]["input_ids"] for index in chosen])
    positions = torch.stack([dataset[int(index)]["position_ids"] for index in chosen])

    tokenizer = AutoTokenizer.from_pretrained(
        metadata["model_id"], revision=metadata["model_revision"]
    )
    labels = object_labels(tokenizer, blocks, args)

    specifications = [("original", None)]
    for entry in args.checkpoint:
        name, _, path = entry.partition("=")
        specifications.append((name, Path(path)))

    states: dict[str, dict[int, torch.Tensor]] = {}
    for name, path in specifications:
        if path is None:
            model = AutoModelForCausalLM.from_pretrained(
                metadata["model_id"], revision=metadata["model_revision"],
                dtype=dtype, low_cpu_mem_usage=True,
            ).to(args.device)
            model.requires_grad_(False)
        else:
            model, _, _ = load_stage1_checkpoint(
                path, device=args.device, dtype=dtype
            )
        model.eval()
        with ResidualCapture(model, layers) as capture:
            states[name] = collect_states(
                model, capture, blocks, positions, layers, args
            )
        del model
        torch.cuda.empty_cache()

    names = [name for name, _ in specifications]
    flat_labels = {key: value.reshape(-1, *value.shape[2:])
                   for key, value in labels.items()}
    report = {
        "schema_version": 1,
        "token_blocks": str(args.token_blocks),
        "blocks": int(args.blocks),
        "positions": int(blocks.numel()),
        "layers": layers,
        "models": names,
        "object_heuristic": {
            "minimum_mentions": args.minimum_mentions,
            "entities_per_block": args.entities_per_block,
            "labelled_mentions": int((flat_labels["entity"] >= 0).sum()),
        },
        "cka": {}, "geometry": {}, "probes": {}, "knn_entity_purity": {},
        "oracle_information": False,
        "candidate_privileged_information": False,
        "cross_project_information": False,
    }

    for layer in layers:
        key = str(layer)
        report["cka"][key] = {}
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                pair = f"{names[left]}|{names[right]}"
                report["cka"][key][pair] = linear_cka(
                    states[names[left]][layer], states[names[right]][layer]
                )
        report["geometry"][key] = {
            name: {
                "effective_rank": effective_rank(
                    covariance_eigenvalues(states[name][layer])
                ),
                "mean_pairwise_cosine": mean_pairwise_cosine(states[name][layer]),
            }
            for name in names
        }

        report["probes"][key] = {}
        report["knn_entity_purity"][key] = {}
        for name in names:
            flat = states[name][layer].reshape(-1, states[name][layer].shape[-1])
            mention = flat_labels["entity"] >= 0
            repeated = flat_labels["recency"] >= 0
            probe_generator = torch.Generator().manual_seed(args.seed + 7)
            shuffled = flat_labels["given_new"].clone()
            valid = shuffled[mention]
            shuffled[mention] = valid[torch.randperm(len(valid), generator=probe_generator)]
            report["probes"][key][name] = {
                "given_new": fit_probe(
                    flat, flat_labels["given_new"].clamp_min(0), task="binary",
                    generator=torch.Generator().manual_seed(args.seed + 1),
                    args=args, mask=mention,
                ),
                "given_new_shuffled_control": fit_probe(
                    flat, shuffled.clamp_min(0), task="binary",
                    generator=torch.Generator().manual_seed(args.seed + 1),
                    args=args, mask=mention,
                ),
                "prefix_object_set": fit_probe(
                    flat, flat_labels["prefix_set"], task="multilabel",
                    generator=torch.Generator().manual_seed(args.seed + 2),
                    args=args,
                ),
                "mention_recency": fit_probe(
                    flat, flat_labels["recency"].float().clamp_min(1).log(),
                    task="regression",
                    generator=torch.Generator().manual_seed(args.seed + 3),
                    args=args, mask=repeated,
                ),
            }
            report["knn_entity_purity"][key][name] = knn_entity_purity(
                flat, flat_labels["entity"],
                generator=torch.Generator().manual_seed(args.seed + 4),
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

    if args.state_dump is not None:
        dump_generator = torch.Generator().manual_seed(args.seed + 5)
        mention = (flat_labels["entity"] >= 0).nonzero().flatten()
        index = mention[torch.randperm(len(mention), generator=dump_generator)
                        [:args.dump_positions]]
        payload = {
            "index": index,
            "entity": flat_labels["entity"][index],
            "given_new": flat_labels["given_new"][index],
            "token_id": blocks.reshape(-1)[index],
            "layers": torch.tensor(layers),
        }
        for name in names:
            for layer in layers:
                flat = states[name][layer].reshape(-1, states[name][layer].shape[-1])
                payload[f"{name}@{layer}"] = flat[index]
        args.state_dump.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.state_dump)


if __name__ == "__main__":
    main()
