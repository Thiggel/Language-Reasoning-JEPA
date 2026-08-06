"""Frozen linear probes for pooled sentence-boundary representations.

Symbolic annotations are used only after training as diagnostics.  Train and
test problems use disjoint procedural seeds; states from one problem never
appear in both probe splits.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.semantic_lm import (
    SemanticBoundaryLMDataset, collate_semantic_lm,
)
from textjepa.models.pooled_sentence_jepa import PooledSentenceJEPA


def dependency_depth(problem, index, cache):
    if index not in cache:
        parents = problem.vars[index].parents
        cache[index] = 0 if not parents else 1 + max(
            dependency_depth(problem, parent, cache) for parent in parents
        )
    return cache[index]


@torch.no_grad()
def extract(model, vocab, cfg, seed, examples, batch_size, device):
    dataset = SemanticBoundaryLMDataset(
        vocab, size=examples, seed=seed, boundary_mode="semantic",
        modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range),
        leaf_prob=cfg.data.leaf_prob, steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=partial(collate_semantic_lm, pad_id=vocab.pad_id),
    )
    features = {"pooled": [], "backbone": [], "previous_token": []}
    labels = {key: [] for key in (
        "op", "value", "var_idx", "parent_1", "parent_2",
        "parent_value_1", "parent_value_2", "necessary", "depth",
        "query_idx", "answer", "remaining", "resolved_n", "n_vars",
        "n_necessary", "solved", "resolved_set", "ancestor_set",
        "feasible_set", "resolved_values",
    )}
    offset = 0
    max_vars = int(cfg.data.n_vars_range[1])
    for batch in loader:
        tokens = batch["tokens"].to(device)
        pooled = model.state_encoder(tokens)
        backbone = model.state_encoder.backbone(tokens)
        for row in range(len(tokens)):
            item_index = offset + row
            raw = dataset.igsm[item_index]
            problem, _ = dataset.igsm.problem(item_index)
            env = SymbolicEnv(problem)
            depth_cache = {}
            sentence_ends = batch["sentence_ends"][row]
            sentence_ends = sentence_ends[sentence_ends.ge(0)].tolist()
            prompt = int(batch["prompt_len"][row])
            for step_index, end in enumerate(sentence_ends):
                position = prompt + int(end) - 1
                action = raw["var_idx"][step_index]
                env.step(action)
                variable = problem.vars[action]
                parents = variable.parents
                features["pooled"].append(pooled[row, position].cpu())
                features["backbone"].append(backbone[row, position].cpu())
                token_feature = torch.zeros(len(vocab))
                previous_position = max(prompt, position - 1)
                token_feature[int(tokens[row, previous_position])] = 1.0
                features["previous_token"].append(token_feature)
                labels["op"].append(raw["op"][step_index])
                labels["value"].append(raw["value"][step_index])
                labels["var_idx"].append(action)
                labels["parent_1"].append(parents[0] if parents else max_vars)
                labels["parent_2"].append(parents[1] if parents else max_vars)
                labels["parent_value_1"].append(
                    problem.values[parents[0]] if parents else problem.modulus
                )
                labels["parent_value_2"].append(
                    problem.values[parents[1]] if parents else problem.modulus
                )
                labels["necessary"].append(int(action in problem.query_ancestors))
                labels["depth"].append(dependency_depth(
                    problem, action, depth_cache
                ))
                labels["query_idx"].append(problem.query)
                labels["answer"].append(problem.answer)
                labels["remaining"].append(env.remaining_necessary())
                labels["resolved_n"].append(len(env.resolved))
                labels["n_vars"].append(len(problem.vars))
                labels["n_necessary"].append(problem.n_necessary_steps)
                labels["solved"].append(int(env.solved))
                resolved = np.zeros(max_vars, dtype=np.int64)
                ancestors = np.zeros(max_vars, dtype=np.int64)
                feasible = np.zeros(max_vars, dtype=np.int64)
                values = np.full(max_vars, problem.modulus, dtype=np.int64)
                for index in env.resolved_set:
                    resolved[index] = 1
                    values[index] = problem.values[index]
                for index in problem.query_ancestors:
                    ancestors[index] = 1
                for index in env.feasible_actions():
                    feasible[index] = 1
                labels["resolved_set"].append(resolved)
                labels["ancestor_set"].append(ancestors)
                labels["feasible_set"].append(feasible)
                labels["resolved_values"].append(values)
        offset += len(tokens)
    return (
        {key: torch.stack(value).numpy() for key, value in features.items()},
        {key: np.asarray(value) for key, value in labels.items()},
    )


def categorical_probe(train_x, train_y, test_x, test_y):
    unique, train_counts = np.unique(train_y, return_counts=True)
    if len(unique) < 2:
        accuracy = float((test_y == unique[0]).mean())
        _, counts = np.unique(test_y, return_counts=True)
        return {"accuracy": accuracy, "majority": float(counts.max() / counts.sum())}
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(max_iter=1000, C=1.0).fit(
        scaler.transform(train_x), train_y
    )
    accuracy = float(model.score(scaler.transform(test_x), test_y))
    _, counts = np.unique(test_y, return_counts=True)
    return {"accuracy": accuracy, "majority": float(counts.max() / counts.sum())}


def multilabel_probe(train_x, train_y, test_x, test_y):
    scaler = StandardScaler().fit(train_x)
    train_scaled, test_scaled = scaler.transform(train_x), scaler.transform(test_x)
    probability = np.zeros_like(test_y, dtype=np.float64)
    for column in range(train_y.shape[1]):
        unique = np.unique(train_y[:, column])
        if len(unique) < 2:
            probability[:, column] = float(unique[0])
        else:
            classifier = LogisticRegression(max_iter=1000, C=1.0).fit(
                train_scaled, train_y[:, column]
            )
            probability[:, column] = classifier.predict_proba(test_scaled)[:, 1]
    prediction = probability >= 0.5
    valid_columns = np.where(
        (test_y.sum(0) > 0) & (test_y.sum(0) < len(test_y))
    )[0]
    auroc = roc_auc_score(
        test_y[:, valid_columns], probability[:, valid_columns], average="macro"
    ) if len(valid_columns) else float("nan")
    return {
        "element_accuracy": float((prediction == test_y).mean()),
        "macro_auroc": float(auroc),
        "exact_set_accuracy": float((prediction == test_y).all(1).mean()),
    }


def value_table_probe(train_x, train_y, test_x, test_y):
    rows = []
    for column in range(train_y.shape[1]):
        rows.append(categorical_probe(
            train_x, train_y[:, column], test_x, test_y[:, column]
        ))
    return {
        "mean_variable_accuracy": float(np.mean([row["accuracy"] for row in rows])),
        "mean_majority": float(np.mean([row["majority"] for row in rows])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--test-examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-seed", type=int, default=310003)
    parser.add_argument("--test-seed", type=int, default=410009)
    args = parser.parse_args()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = PooledSentenceJEPA(
        len(vocab), vocab.pad_id, period_id=vocab.token_to_id["."],
        question_id=vocab.token_to_id["?"], **cfg.model,
    ).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    train_x, train_y = extract(
        model, vocab, cfg, args.train_seed, args.train_examples,
        args.batch_size, args.device,
    )
    test_x, test_y = extract(
        model, vocab, cfg, args.test_seed, args.test_examples,
        args.batch_size, args.device,
    )
    categorical = [
        "op", "value", "var_idx", "parent_1", "parent_2",
        "parent_value_1", "parent_value_2", "necessary", "depth",
        "query_idx", "answer", "remaining", "resolved_n", "n_vars",
        "n_necessary", "solved",
    ]
    result = {
        "probe_only_symbolic_labels": True,
        "problem_disjoint_train_test": True,
        "train_examples": args.train_examples,
        "test_examples": args.test_examples,
        "features": {},
    }
    for feature_name in train_x:
        result["features"][feature_name] = {
            label: categorical_probe(
                train_x[feature_name], train_y[label],
                test_x[feature_name], test_y[label],
            ) for label in categorical
        }
        for label in ("resolved_set", "ancestor_set", "feasible_set"):
            result["features"][feature_name][label] = multilabel_probe(
                train_x[feature_name], train_y[label],
                test_x[feature_name], test_y[label],
            )
        result["features"][feature_name]["resolved_values"] = value_table_probe(
            train_x[feature_name], train_y["resolved_values"],
            test_x[feature_name], test_y["resolved_values"],
        )
    scaler = StandardScaler().fit(train_x["previous_token"])
    regressor = Ridge(alpha=1.0).fit(
        scaler.transform(train_x["previous_token"]), train_x["pooled"]
    )
    result["pooled_variance_explained_by_previous_token"] = float(
        regressor.score(
            scaler.transform(test_x["previous_token"]), test_x["pooled"]
        )
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
