"""Comprehensive grouped linear probes for shared token-hierarchy states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score

from textjepa.data.igsm.dataset import OP_LABELS, build_vocab
from textjepa.data.lm import LMDataset
from textjepa.models.token_hierarchy_v2 import MultilevelTokenHierarchyJEPA


def split(groups, fold):
    test = groups % 5 == fold
    return ~test, test


def classify(x, y, groups, fold=0):
    train, test = split(groups, fold)
    if train.sum() < 20 or test.sum() < 5 or len(np.unique(y[train])) < 2:
        return {"accuracy": float("nan"), "balanced_accuracy": float("nan"),
                "majority": float("nan"), "n": int(test.sum())}
    model = RidgeClassifier(alpha=1.0).fit(x[train], y[train])
    majority = np.bincount(y[train].astype(int)).argmax()
    prediction = model.predict(x[test])
    return {
        "accuracy": float(accuracy_score(y[test], prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test], prediction)),
        "majority": float((y[test] == majority).mean()), "n": int(test.sum()),
    }


def regress(x, y, groups, fold=1):
    train, test = split(groups, fold)
    if train.sum() < 20 or test.sum() < 5:
        return {"r2": float("nan"), "n": int(test.sum())}
    model = Ridge(alpha=1.0).fit(x[train], y[train])
    return {"r2": float(r2_score(y[test], model.predict(x[test]))), "n": int(test.sum())}


def mean_metric(rows, key):
    values = [row[key] for row in rows if np.isfinite(row.get(key, np.nan))]
    return float(np.mean(values)) if values else float("nan")


def dependency_depths(problem):
    depths = []
    for variable in problem.vars:
        depths.append(
            0 if variable.is_leaf
            else 1 + max(depths[parent] for parent in variable.parents)
        )
    return np.asarray(depths, dtype=int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=512)
    args = parser.parse_args()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
    ).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    dataset = LMDataset(
        vocab, size=args.examples, seed=cfg.data.val_seed,
        modulus=cfg.data.modulus, n_vars_range=tuple(cfg.data.n_vars_range),
        leaf_prob=cfg.data.leaf_prob, steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )
    max_vars = int(cfg.data.n_vars_range[1])
    prompt_x, prompt_group = [], []
    step_x, step_group = [], []
    labels = {key: [] for key in (
        "n_vars", "n_necessary", "query", "answer", "step_index",
        "current_var", "current_value", "current_op", "necessary",
        "remaining", "resolved_count", "next_var", "next_value", "next_op",
    )}
    resolved_matrix, feasible_matrix, unresolved_necessary_matrix = [], [], []
    ancestor_matrix, all_value_matrix, var_op_matrix, var_mask = [], [], [], []
    prompt_value, prompt_op, prompt_ancestor, prompt_leaf = [], [], [], []
    prompt_depth, prompt_mask = [], []
    prompt_adjacency = []
    final_x, final_group = [], []
    with torch.no_grad():
        for index in range(args.examples):
            item = dataset.igsm[index]
            problem, _ = dataset.igsm.problem(index)
            prompt = [token for sentence in item["prompt"] for token in sentence]
            steps = [[token for token in sentence] for sentence in item["steps"]]
            tokens = prompt + [token for sentence in steps for token in sentence]
            states = model.encoder(torch.tensor([tokens], device=args.device))[0]
            prompt_x.append(states[len(prompt) - 1].cpu().numpy())
            prompt_group.append(index)
            p_mask = np.arange(max_vars) < len(problem.vars)
            p_values = np.zeros(max_vars, dtype=int)
            p_ops = np.full(max_vars, len(OP_LABELS), dtype=int)
            p_ancestor = np.zeros(max_vars, dtype=int)
            p_leaf = np.zeros(max_vars, dtype=int)
            p_depth = np.zeros(max_vars, dtype=int)
            depths = dependency_depths(problem)
            for v, variable in enumerate(problem.vars):
                p_values[v] = problem.values[v]
                p_ops[v] = OP_LABELS[variable.op]
                p_ancestor[v] = int(v in problem.query_ancestors)
                p_leaf[v] = int(variable.is_leaf)
                p_depth[v] = depths[v]
            prompt_value.append(p_values); prompt_op.append(p_ops)
            prompt_ancestor.append(p_ancestor); prompt_leaf.append(p_leaf)
            prompt_depth.append(p_depth); prompt_mask.append(p_mask)
            adjacency = np.zeros((max_vars, max_vars), dtype=int)
            for child, variable in enumerate(problem.vars):
                for parent in range(len(problem.vars)):
                    adjacency[child, parent] = int(parent in variable.parents)
            prompt_adjacency.append(adjacency)
            endpoints, cursor = [], len(prompt)
            for sentence in steps:
                cursor += len(sentence)
                endpoints.append(cursor - 1)
            resolved = set()
            for step_index, endpoint in enumerate(endpoints):
                state = states[endpoint].cpu().numpy()
                step_x.append(state); step_group.append(index)
                current_var = int(item["var_idx"][step_index])
                resolved.add(current_var)
                next_index = min(step_index + 1, len(endpoints) - 1)
                values = {
                    "n_vars": len(problem.vars),
                    "n_necessary": problem.n_necessary_steps,
                    "query": problem.query,
                    "answer": problem.answer,
                    "step_index": step_index,
                    "current_var": current_var,
                    "current_value": int(item["value"][step_index]),
                    "current_op": int(item["op"][step_index]),
                    "necessary": int(item["necessary"][step_index]),
                    "remaining": int(item["remaining"][step_index]),
                    "resolved_count": int(item["resolved_n"][step_index]),
                    "next_var": int(item["var_idx"][next_index]),
                    "next_value": int(item["value"][next_index]),
                    "next_op": int(item["op"][next_index]),
                }
                for key, value in values.items(): labels[key].append(value)
                mask = np.arange(max_vars) < len(problem.vars)
                resolved_row = np.array([int(v in resolved) for v in range(max_vars)])
                feasible = np.array([
                    int(v not in resolved and all(p in resolved for p in problem.vars[v].parents))
                    if v < len(problem.vars) else 0
                    for v in range(max_vars)
                ])
                ancestors = np.array([
                    int(v in problem.query_ancestors) for v in range(max_vars)
                ])
                unresolved_necessary = ancestors * (1 - resolved_row)
                all_values = np.zeros(max_vars, dtype=int)
                operations = np.full(max_vars, len(OP_LABELS), dtype=int)
                for v, variable in enumerate(problem.vars):
                    all_values[v] = problem.values[v]
                    operations[v] = OP_LABELS[variable.op]
                resolved_matrix.append(resolved_row)
                feasible_matrix.append(feasible)
                unresolved_necessary_matrix.append(unresolved_necessary)
                ancestor_matrix.append(ancestors); all_value_matrix.append(all_values)
                var_op_matrix.append(operations); var_mask.append(mask)
            final_x.append(states[endpoints[-1]].cpu().numpy())
            final_group.append(index)
    px, pg = np.asarray(prompt_x), np.asarray(prompt_group)
    sx, sg = np.asarray(step_x), np.asarray(step_group)
    fx, fg = np.asarray(final_x), np.asarray(final_group)
    y = {key: np.asarray(value) for key, value in labels.items()}
    results = {
        "prompt": {
            "n_vars": classify(px, np.array([len(dataset.igsm.problem(i)[0].vars) for i in pg]), pg),
            "n_necessary": classify(px, np.array([dataset.igsm.problem(i)[0].n_necessary_steps for i in pg]), pg),
            "query_variable": classify(px, np.array([dataset.igsm.problem(i)[0].query for i in pg]), pg),
            "answer": classify(px, np.array([dataset.igsm.problem(i)[0].answer for i in pg]), pg),
        },
        "step": {
            key: classify(sx, y[key], sg, fold=(i % 5))
            for i, key in enumerate((
                "n_vars", "n_necessary", "query", "answer", "current_var",
                "current_value", "current_op", "necessary", "remaining",
                "resolved_count", "next_var", "next_value", "next_op",
            ))
        },
        "continuous": {
            "step_index_r2": regress(sx, y["step_index"], sg),
            "remaining_r2": regress(sx, y["remaining"], sg, fold=2),
            "resolved_count_r2": regress(sx, y["resolved_count"], sg, fold=3),
        },
        "final": {
            "answer": classify(
                fx, np.array([dataset.igsm.problem(i)[0].answer for i in fg]), fg
            )
        },
    }
    prompt_matrices = {
        "variable_value": np.asarray(prompt_value),
        "variable_operation": np.asarray(prompt_op),
        "query_ancestor_membership": np.asarray(prompt_ancestor),
        "leaf_membership": np.asarray(prompt_leaf),
        "dependency_depth": np.asarray(prompt_depth),
    }
    p_mask = np.asarray(prompt_mask)
    results["prompt_graph"] = {}
    for name, matrix in prompt_matrices.items():
        rows = []
        for variable in range(max_vars):
            valid = p_mask[:, variable]
            if valid.sum() >= 30:
                rows.append(classify(
                    px[valid], matrix[valid, variable], pg[valid],
                    fold=variable % 5,
                ))
        results["prompt_graph"][name] = {
            "mean_accuracy": mean_metric(rows, "accuracy"),
            "mean_balanced_accuracy": mean_metric(rows, "balanced_accuracy"),
            "mean_majority": mean_metric(rows, "majority"),
            "variables": rows,
        }
    edge_rows = []
    adjacency = np.asarray(prompt_adjacency)
    for child in range(max_vars):
        for parent in range(child):
            valid = p_mask[:, child] & p_mask[:, parent]
            if valid.sum() >= 30:
                edge_rows.append(classify(
                    px[valid], adjacency[valid, child, parent], pg[valid],
                    fold=(child + parent) % 5,
                ))
    results["prompt_graph"]["parent_edge_membership"] = {
        "mean_accuracy": mean_metric(edge_rows, "accuracy"),
        "mean_balanced_accuracy": mean_metric(edge_rows, "balanced_accuracy"),
        "mean_majority": mean_metric(edge_rows, "majority"),
        "edges": edge_rows,
    }
    matrices = {
        "resolved_membership": np.asarray(resolved_matrix),
        "known_variable_value": np.asarray(all_value_matrix),
        "feasible_variable_membership": np.asarray(feasible_matrix),
        "unresolved_necessary_membership": np.asarray(unresolved_necessary_matrix),
        "query_ancestor_membership": np.asarray(ancestor_matrix),
        "all_variable_value": np.asarray(all_value_matrix),
        "variable_operation": np.asarray(var_op_matrix),
    }
    mask = np.asarray(var_mask)
    for name, matrix in matrices.items():
        per_variable = []
        for variable in range(max_vars):
            valid = mask[:, variable]
            if name == "known_variable_value":
                valid = valid & (np.asarray(resolved_matrix)[:, variable] == 1)
            if valid.sum() < 30:
                continue
            per_variable.append(classify(
                sx[valid], matrix[valid, variable], sg[valid], fold=variable % 5
            ))
        results[name] = {
            "mean_accuracy": mean_metric(per_variable, "accuracy"),
            "mean_balanced_accuracy": mean_metric(
                per_variable, "balanced_accuracy"
            ),
            "mean_majority": mean_metric(per_variable, "majority"),
            "variables": per_variable,
        }
    dest = Path(args.ckpt).parent / "symbolic_linear_probes.json"
    dest.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
