"""Fit non-symbolic goal-advantage heads on a frozen token hierarchy JEPA.

Targets are changes in latent distance to the encoded terminal state.  No
symbolic feasibility, proof distance, auxiliary language model, or answer
label is used.  Primitive candidates are vocabulary tokens.  Macro proposals
come from observed chunks, the learned conditional prior, or perturbations of
observed chunks.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from plan_token_hierarchy_oracle_cem import (
    build_banks, load_model, make_dataset,
)
from textjepa.models.heads import MacroValueHead, SubgoalActionHead
from textjepa.planning.token_cem import latent_l1


def pair_loss(score, target):
    delta = target[:, :, None] - target[:, None, :]
    valid = delta.abs() > 1e-6
    margin = (score[:, :, None] - score[:, None, :]) * delta.sign()
    return F.softplus(-margin)[valid].mean() if valid.any() else score.sum() * 0


def metrics(score, target):
    delta_t = target[:, :, None] - target[:, None, :]
    delta_s = score[:, :, None] - score[:, None, :]
    valid = delta_t.abs() > 1e-6
    pair = ((delta_t.sign() == delta_s.sign()) & valid).sum() / valid.sum().clamp_min(1)
    chosen = score.argmax(1)
    best = target.max(1).values
    selected = target.gather(1, chosen[:, None]).squeeze(1)
    x, y = score.flatten(), target.flatten()
    x, y = x - x.mean(), y - y.mean()
    corr = (x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-8)
    return {
        "mae": float((score - target).abs().mean()),
        "correlation": float(corr),
        "pair_accuracy": float(pair),
        "top1_optimal": float((selected >= best - 1e-6).float().mean()),
        "top1_regret": float((best - selected).mean()),
    }


def macro_outcomes(model, level_index, state_history, action_history, candidates):
    n = len(candidates)
    states = state_history.expand(n, -1, -1)
    old = action_history.expand(n, -1, -1)
    actions = torch.cat([old, candidates[:, None]], 1)
    return model.levels[level_index].predictor(states, actions)[:, -1]


def proposals(module, bank, state, reference, kind, count):
    if kind in {"codebook", "mixed"}:
        k = count if kind == "codebook" else count // 2
        ids = torch.cdist(state[None], bank["states"]).squeeze(0).topk(
            min(k, len(bank["states"])), largest=False
        ).indices
        codes = bank["actions"][ids]
        raw = bank["raw_ids"][ids]
    else:
        codes = reference.new_zeros(0, reference.shape[-1])
        raw = None
    need = count - len(codes)
    if need:
        if kind in {"prior", "mixed"}:
            mu, logvar = module.action.prior_params(state[None])
            extra = mu + (0.5 * logvar).exp() * torch.randn(
                need, mu.shape[-1], device=state.device
            )
        elif kind == "perturb":
            scale = bank["actions"].std(0, unbiased=False).clamp_min(0.05)
            extra = reference + torch.randn(
                need, reference.shape[-1], device=state.device
            ) * scale
        else:
            raise ValueError(kind)
        codes = torch.cat([codes, extra], 0)
    return codes[:count], raw


@torch.no_grad()
def collect_macro(model, cfg, vocab, device, args):
    level_index = len(model.levels) - 1 if args.level < 0 else args.level
    banks = build_banks(model, cfg, vocab, device, args.bank_examples, args.bank_size)
    bank = banks[level_index]
    ds = make_dataset(cfg, vocab, args.examples * 2, cfg.data.val_seed + 919)
    states, goals, actions, targets, recall = [], [], [], [], []
    for i in range(len(ds)):
        item = ds[i]
        tokens = torch.tensor([item["tokens"]], device=device)
        prompt_len = torch.tensor([item["prompt_len"]], device=device)
        out = model(tokens, prompt_len)
        level = out["levels"][level_index]
        count = int(level["valid"][0].sum())
        if not count:
            continue
        teacher = model.teacher(tokens)[:, item["prompt_len"] - 1:]
        goal = model.lift_state_path(
            teacher, through_level=level_index, teacher=True
        )[level_index][:, -1]
        for j in range(count):
            state = level["prev"][0, j]
            reference = level["codes"][0, j]
            candidate, raw = proposals(
                model.levels[level_index], bank, state, reference,
                args.proposal, args.candidates
            )
            state_history = level["prev"][:, :j + 1]
            action_history = level["codes"][:, :j]
            outcome = macro_outcomes(
                model, level_index, state_history, action_history, candidate
            )
            if args.outcome_source == "teacher":
                if args.proposal != "codebook" or args.horizon != 1:
                    raise ValueError(
                        "teacher outcomes currently require one-step codebook proposals"
                    )
                token_start = int(level["phase_offsets"][0]) + j * level["span"]
                prefix = torch.tensor(
                    item["tokens"][:item["prompt_len"] + token_start],
                    device=device,
                )
                counterfactual = torch.cat([
                    prefix[None].expand(len(candidate), -1), raw
                ], 1)
                base = model.teacher(counterfactual)[:, item["prompt_len"] - 1:]
                outcome = model.lift_state_path(
                    base, through_level=level_index, teacher=True
                )[level_index][:, -1]
            current = state[None].expand_as(outcome)
            target = latent_l1(current, goal.expand_as(current)) - latent_l1(
                outcome, goal.expand_as(outcome)
            )
            # Approximate optimal continuation: at each later step greedily
            # choose the best of a fresh conditional proposal set.
            cur, hist_s = outcome, torch.cat([
                state_history.expand(len(candidate), -1, -1), outcome[:, None]
            ], 1)
            hist_a = torch.cat([
                action_history.expand(len(candidate), -1, -1), candidate[:, None]
            ], 1)
            for _ in range(1, args.horizon):
                branch_outcomes = []
                branch_codes = []
                for b in range(len(candidate)):
                    options, _ = proposals(
                        model.levels[level_index], bank, cur[b], candidate[b], args.proposal,
                        args.continuation_candidates,
                    )
                    predicted = macro_outcomes(
                        model, level_index, hist_s[b:b + 1],
                        hist_a[b:b + 1], options,
                    )
                    best = latent_l1(
                        predicted, goal.expand_as(predicted)
                    ).argmin()
                    branch_outcomes.append(predicted[best])
                    branch_codes.append(options[best])
                cur = torch.stack(branch_outcomes)
                chosen = torch.stack(branch_codes)
                hist_s = torch.cat([hist_s, cur[:, None]], 1)
                hist_a = torch.cat([hist_a, chosen[:, None]], 1)
            if args.horizon > 1:
                target = latent_l1(current, goal.expand_as(current)) - latent_l1(
                    cur, goal.expand_as(cur)
                )
            states.append(state.cpu()); goals.append(goal[0].cpu())
            actions.append(candidate.cpu()); targets.append(target.cpu())
            recall.append(float(torch.cdist(
                reference[None], candidate
            ).amin()))
            if len(states) >= args.examples:
                return tuple(map(torch.stack, (states, goals, actions, targets))), recall
    return tuple(map(torch.stack, (states, goals, actions, targets))), recall


@torch.no_grad()
def collect_primitive(model, cfg, vocab, device, args):
    ds = make_dataset(cfg, vocab, args.examples * 2, cfg.data.val_seed + 1217)
    states, goals, actions, targets, recall = [], [], [], [], []
    allowed = [i for i in range(len(vocab)) if i not in (
        vocab.pad_id, vocab.token_to_id[vocab.UNK]
    )]
    for i in range(len(ds)):
        item = ds[i]
        ids = item["tokens"]
        prompt = item["prompt_len"]
        full = torch.tensor([ids], device=device)
        online = model.encoder(full)
        teacher = model.teacher(full)
        goal = teacher[:, -1]
        positions = list(range(prompt - 1, len(ids) - 1))
        random.shuffle(positions)
        for pos in positions[:2]:
            chosen = allowed if args.candidates >= len(allowed) else random.sample(
                allowed, args.candidates
            )
            candidate_ids = torch.tensor(chosen, device=device)
            candidate = model.token_action(candidate_ids)
            state_hist = online[:, prompt - 1:pos + 1]
            old_ids = torch.tensor(
                [ids[prompt:pos + 1]], device=device, dtype=torch.long
            )
            old_actions = model.token_action(old_ids)
            outcome = model.low_predictor(
                state_hist.expand(len(chosen), -1, -1),
                torch.cat([
                    old_actions.expand(len(chosen), -1, -1), candidate[:, None]
                ], 1),
            )[:, -1]
            current = online[:, pos].expand_as(outcome)
            target = latent_l1(current, goal.expand_as(current)) - latent_l1(
                outcome, goal.expand_as(outcome)
            )
            cur = outcome
            hist_s = torch.cat([
                state_hist.expand(len(chosen), -1, -1), cur[:, None]
            ], 1)
            hist_a = torch.cat([
                old_actions.expand(len(chosen), -1, -1), candidate[:, None]
            ], 1)
            all_actions = model.token_action(torch.tensor(allowed, device=device))
            for _ in range(1, args.horizon):
                next_states = []
                for b in range(len(chosen)):
                    pred = model.low_predictor(
                        hist_s[b:b + 1].expand(len(allowed), -1, -1),
                        torch.cat([
                            hist_a[b:b + 1].expand(len(allowed), -1, -1),
                            all_actions[:, None],
                        ], 1),
                    )[:, -1]
                    next_states.append(pred[latent_l1(
                        pred, goal.expand_as(pred)
                    ).argmin()])
                cur = torch.stack(next_states)
                hist_s = torch.cat([hist_s, cur[:, None]], 1)
            if args.horizon > 1:
                target = latent_l1(current, goal.expand_as(current)) - latent_l1(
                    cur, goal.expand_as(cur)
                )
            states.append(online[0, pos].cpu()); goals.append(goal[0].cpu())
            actions.append(candidate.cpu()); targets.append(target.cpu())
            ref = ids[pos + 1]
            recall.append(float(ref in chosen))
            if len(states) >= args.examples:
                return tuple(map(torch.stack, (states, goals, actions, targets))), recall
    return tuple(map(torch.stack, (states, goals, actions, targets))), recall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--device", default="cuda:0")
    p.add_argument("--scope", choices=["primitive", "macro"], required=True)
    p.add_argument("--proposal", choices=["codebook", "prior", "mixed", "perturb"], default="codebook")
    p.add_argument("--loss", choices=["regression", "combined"], default="combined")
    p.add_argument("--level", type=int, default=-1); p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--outcome-source", choices=["predicted", "teacher"], default="predicted")
    p.add_argument("--examples", type=int, default=256); p.add_argument("--candidates", type=int, default=16)
    p.add_argument("--continuation-candidates", type=int, default=8)
    p.add_argument("--bank-examples", type=int, default=256); p.add_argument("--bank-size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--rank-weight", type=float, default=1.0); p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=73)
    args = p.parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model, vocab, cfg = load_model(args.ckpt, args.device)
    data, recall = (collect_primitive if args.scope == "primitive" else collect_macro)(
        model, cfg, vocab, args.device, args
    )
    state, goal, action, target = [x.to(args.device) for x in data]
    n = len(state); split = max(1, int(0.8 * n))
    head = (SubgoalActionHead(model.d_model, model.d_action) if args.scope == "primitive" else
            MacroValueHead(model.d_model, action.shape[-1])).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    mean, std = target[:split].mean(), target[:split].std().clamp_min(1e-4)
    normalized = (target - mean) / std
    for _ in range(args.epochs):
        score = head(state[:split, None].expand(-1, action.shape[1], -1),
                     goal[:split, None].expand(-1, action.shape[1], -1), action[:split])
        loss = F.mse_loss(score, normalized[:split])
        if args.loss == "combined": loss = loss + args.rank_weight * pair_loss(score, normalized[:split])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        score = head(state[split:, None].expand(-1, action.shape[1], -1),
                     goal[split:, None].expand(-1, action.shape[1], -1), action[split:])
        score = score * std + mean
    result = {
        "checkpoint": args.ckpt, "scope": args.scope, "proposal": args.proposal,
        "target": f"non_symbolic_{args.outcome_source}_latent_goal_distance_change", "horizon": args.horizon,
        "loss": args.loss, "train_anchors": split, "validation_anchors": n - split,
        ("reference_proposal_recall" if args.scope == "primitive" else
         "reference_proposal_distance"): float(np.mean(recall)),
        "validation": metrics(score, target[split:]), "args": vars(args),
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "result": result}, out.with_suffix(".pt"))
    out.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
