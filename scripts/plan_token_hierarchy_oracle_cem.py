"""Oracle-goal, no-LM hierarchical planning for token hierarchy v2.

Continuous CEM plans macro-actions; categorical CEM plans primitive token IDs.
No language model, symbolic grammar, or reference-token proposal enters search.
The symbolic environment is used only after execution to measure success.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import step_sentence
from textjepa.data.lm import LMDataset, collate_lm
from textjepa.models.token_hierarchy_v2 import MultilevelTokenHierarchyJEPA
from textjepa.models.heads import MacroValueHead, SubgoalActionHead
from textjepa.planning.token_cem import (
    CEMResult,
    batched_categorical_min_cost,
    categorical_cem,
    continuous_cem,
    gaussian_mixture_nll,
    latent_l1,
)
from textjepa.planning.token_hierarchy import (
    feedback_levels_to_invalidate,
    macro_codes,
    remaining_to_boundary,
)


def load_model(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, vocab, cfg


def make_dataset(cfg, vocab, size, seed):
    return LMDataset(
        vocab, size=size, seed=seed, modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range), leaf_prob=cfg.data.leaf_prob,
        steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )


@torch.no_grad()
def build_banks(model, cfg, vocab, device, examples, max_codes):
    ds = make_dataset(cfg, vocab, examples, cfg.data.train_seed + 104729)
    loader = DataLoader(
        ds, batch_size=16, collate_fn=partial(collate_lm, pad_id=vocab.pad_id)
    )
    stores = [dict(states=[], actions=[], support=[], raw_ids=[]) for _ in model.levels]
    for batch in loader:
        out = model(batch["tokens"].to(device), batch["prompt_len"].to(device))
        for index, level in enumerate(out["levels"]):
            valid = level["valid"]
            stores[index]["states"].append(level["prev"][valid].cpu())
            stores[index]["actions"].append(level["codes"][valid].cpu())
            stores[index]["support"].append(level["support_pos"][valid].cpu())
            stores[index]["raw_ids"].append(level["raw_action_ids"][valid].cpu())
        if all(sum(len(x) for x in store["actions"]) >= max_codes for store in stores):
            break
    banks = []
    for store in stores:
        banks.append({
            key: torch.cat(value)[:max_codes].to(device)
            for key, value in store.items()
        })
    return banks


def load_or_build_banks(model, cfg, vocab, device, examples, max_codes, cache_path):
    """Cache observed macro support once per checkpoint and bank specification."""
    path = Path(cache_path) if cache_path else None
    signature = {
        "level_spans": list(model.level_spans),
        "level_dims": list(model.level_dims),
        "examples": int(examples),
        "max_codes": int(max_codes),
        "train_seed": int(cfg.data.train_seed),
        "bank_schema": 2,
    }
    if path is not None and path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("signature") == signature:
            return [
                {key: value.to(device) for key, value in bank.items()}
                for bank in payload["banks"]
            ]
    banks = build_banks(model, cfg, vocab, device, examples, max_codes)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "signature": signature,
            "banks": [
                {key: value.detach().cpu() for key, value in bank.items()}
                for bank in banks
            ],
        }, path)
    return banks


def fit_gmm(actions, components):
    n_components = min(components, max(1, len(actions) // 32))
    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full", reg_covar=1e-4,
        max_iter=100, random_state=0,
    ).fit(actions.detach().cpu().numpy())
    device, dtype = actions.device, actions.dtype
    return (
        torch.tensor(gmm.weights_, device=device, dtype=dtype),
        torch.tensor(gmm.means_, device=device, dtype=dtype),
        torch.tensor(gmm.covariances_, device=device, dtype=dtype),
    )


def histories(model, prefix_states, prompt_len, generated, level_index):
    span = model.level_spans[level_index]
    complete = len(generated) // span
    ids = [prompt_len - 1 + step * span for step in range(complete + 1)]
    state_history = prefix_states[:, ids]
    if complete:
        tokens = torch.tensor([generated[:complete * span]], device=prefix_states.device)
        action_history = macro_codes(model, tokens, through_level=level_index)[level_index]
    else:
        action_history = prefix_states.new_zeros(
            1, 0, model.level_dims[level_index]
        )
    return state_history, action_history


def primitive_history(model, prefix_states, prompt_len, generated):
    state_history = prefix_states[:, prompt_len - 1:prompt_len + len(generated)]
    if generated:
        ids = torch.tensor([generated], device=prefix_states.device)
        actions = model.token_action(ids)
    else:
        actions = prefix_states.new_zeros(1, 0, model.d_action)
    return state_history, actions


def replay(problem, vocab, generated):
    env = SymbolicEnv(problem)
    period = vocab.token_to_id["."]
    start = valid = invalid = 0
    for end, token in enumerate(generated):
        if token != period:
            continue
        sentence = generated[start:end + 1]
        start = end + 1
        chosen = None
        for action in env.feasible_actions():
            if vocab.encode(step_sentence(problem, action)) == sentence:
                chosen = action
                break
        if chosen is None:
            invalid += 1
        else:
            valid += 1
            env.step(chosen)
        if env.solved:
            break
    return env.solved, valid, invalid


class OracleCEMPlanner:
    def __init__(self, model, vocab, banks, gmms, args, ensembles=None):
        self.model, self.vocab = model, vocab
        self.banks, self.gmms, self.args = banks, gmms, args
        self.ensembles = ensembles
        self.device = next(model.parameters()).device
        self.cache = {}
        self.pending = []
        self.records = []
        self.oracle_goal = None
        self.oracle_level_goals = None
        self.advantage_head = None
        self.advantage_scope = None
        if args.advantage_head:
            payload = torch.load(
                args.advantage_head, map_location="cpu", weights_only=False
            )
            self.advantage_scope = payload["result"]["scope"]
            if self.advantage_scope == "primitive":
                head = SubgoalActionHead(model.d_model, model.d_action)
            else:
                head = MacroValueHead(model.d_model, model.level_dims[-1])
            head.load_state_dict(payload["head"])
            self.advantage_head = head.to(self.device).eval()

    def geometric_goal_cost(self, head):
        """Return a CEM cost using only JEPA geometry and its learned energy.

        ``combined`` standardizes both signals within the candidate population
        before adding them.  This avoids treating the arbitrary scale of a
        ranking-trained value head as calibrated metric distance.
        """
        mode = self.args.goal_score
        if mode == "latent_distance":
            return None

        def score(states, goals):
            value = head(states, goals)
            if mode == "learned_value":
                return value
            distance = latent_l1(states, goals)

            def standardized(x):
                return (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-6)

            return standardized(distance) + self.args.value_weight * standardized(value)

        return score

    def lift_low_predictions(self, base_path, predictions, level_index):
        """Map primitive rollout endpoints into a distinct higher state space.

        The observed causal prefix is prepended before re-encoding.  This is
        essential: a distinct level encoder is history-dependent, so encoding
        a candidate endpoint in isolation is not the model used in training.
        """
        n = len(predictions)
        prefix = base_path.expand(n, -1, -1)
        path = torch.cat([prefix, predictions], 1)
        lifted = self.model.lift_state_path(path, through_level=level_index)
        return lifted[level_index][:, -1:]

    def lift_level_predictions(self, level_path, predictions, source_level):
        """Lift a rollout one hierarchy level for recursive subgoal scoring."""
        if source_level + 1 >= len(self.model.levels):
            raise ValueError("top-level rollouts cannot be lifted further")
        n = len(predictions)
        path = torch.cat([
            level_path.expand(n, -1, -1), predictions
        ], 1)
        valid = torch.ones(path.shape[:2], dtype=torch.bool, device=path.device)
        encoder = self.model.levels[source_level + 1].state_encoder
        lifted = encoder(path, valid) if encoder is not None else path
        return lifted[:, -1:]

    def low_rollout(self, start, tokens, state_history, action_history):
        actions = self.model.token_action(tokens)
        return self.model.low_predictor.rollout(
            start.expand(len(tokens), -1), actions,
            state_history=state_history, action_history=action_history,
        )

    def token_plan(self, start, target, horizon, state_history, action_history,
                   reduced=False, target_level=None, base_path=None):
        scale = self.args.reach_budget_scale if reduced else 1.0
        candidates = max(16, int(self.args.token_candidates * scale))
        iterations = max(1, int(self.args.token_iterations * scale))
        mode = self.args.token_proposal
        if mode != "uniform" and self.model.token_prior is None:
            raise ValueError(
                f"token proposal {mode!r} requires a checkpoint trained with token_prior"
            )

        if mode in ("prior_shooting", "prior_greedy"):
            return self.prior_token_plan(
                start, target, horizon, state_history, action_history,
                candidates=(1 if mode == "prior_greedy" else candidates),
                greedy=(mode == "prior_greedy"),
            )

        cost_terms = []
        if mode == "prior_energy":
            def prior_cost(tokens, states):
                starts = torch.cat([
                    start.expand(len(tokens), -1)[:, None], states[:, :-1]
                ], 1)
                logits = self.model.token_prior(starts) / self.args.token_prior_temperature
                nll = -logits.log_softmax(-1).gather(
                    -1, tokens[..., None]
                ).squeeze(-1).mean(1)
                return self.args.token_prior_weight * nll, {
                    "token_prior_nll": nll.detach(),
                }
            cost_terms.append(prior_cost)
        if self.advantage_scope == "primitive":
            def advantage_cost(tokens, states):
                starts = torch.cat([
                    start.expand(len(tokens), -1)[:, None], states[:, :-1]
                ], 1)
                embedded = self.model.token_action(tokens)
                goals = target.expand(len(tokens), -1)[:, None].expand_as(starts)
                advantage = self.advantage_head(starts, goals, embedded).mean(1)
                return -self.args.advantage_weight * advantage, {
                    "distilled_advantage": advantage.detach(),
                }
            cost_terms.append(advantage_cost)
        def extra_cost(tokens, states):
            addition = states.new_zeros(len(tokens))
            diagnostics = {}
            for term in cost_terms:
                value, extra = term(tokens, states)
                addition = addition + value
                diagnostics.update(extra)
            return addition, diagnostics
        transform = None
        if target_level is not None:
            if base_path is None:
                raise ValueError("distinct-level token planning requires base_path")
            transform = lambda states: self.lift_low_predictions(
                base_path, states, target_level
            )
        value_head = (
            self.model.levels[target_level].goal_value
            if target_level is not None else self.model.low_goal_value
        )
        result = categorical_cem(
            lambda ids: self.low_rollout(
                start, ids, state_history, action_history
            ),
            target, horizon, len(self.vocab), candidates=candidates,
            iterations=iterations,
            elites=min(self.args.token_elites, candidates), alpha=self.args.alpha,
            forbidden=(
                self.vocab.pad_id,
                self.vocab.token_to_id[self.vocab.UNK],
            ),
            extra_cost=(extra_cost if cost_terms else None),
            goal_states=transform,
            goal_cost_fn=self.geometric_goal_cost(value_head),
            rollout_batch_size=self.args.cem_rollout_batch_size,
        )
        if target_level is not None:
            # Keep primitive predictions for execution-drift diagnostics.  The
            # transformed states were used only for the goal cost.
            result.states = self.low_rollout(
                start, result.actions[None], state_history, action_history
            )[0]
        return result

    def prior_token_plan(self, start, target, horizon, state_history,
                         action_history, candidates, greedy=False):
        """Autoregressive proposals from the JEPA's state-conditioned prior."""
        n = int(candidates)
        states_hist = state_history.expand(n, -1, -1)
        actions_hist = action_history.expand(n, -1, -1)
        current = start.expand(n, -1)
        predictions, tokens, nlls, entropies = [], [], [], []
        forbidden = (self.vocab.pad_id, self.vocab.token_to_id[self.vocab.UNK])
        for _ in range(horizon):
            logits = self.model.token_prior(current) / self.args.token_prior_temperature
            logits[:, list(forbidden)] = -torch.inf
            if 0 < self.args.token_prior_topk < logits.shape[-1]:
                threshold = logits.topk(self.args.token_prior_topk, -1).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, -torch.inf)
            distribution = torch.distributions.Categorical(logits=logits)
            token = logits.argmax(-1) if greedy else distribution.sample()
            action = self.model.token_action(token)
            actions_hist = torch.cat([actions_hist, action[:, None]], 1)
            current = self.model.low_predictor(states_hist, actions_hist)[:, -1]
            states_hist = torch.cat([states_hist, current[:, None]], 1)
            predictions.append(current)
            tokens.append(token)
            nlls.append(-distribution.log_prob(token))
            entropies.append(distribution.entropy())
        states = torch.stack(predictions, 1)
        token_ids = torch.stack(tokens, 1)
        prior_nll = torch.stack(nlls, 1).mean(1)
        prior_entropy = torch.stack(entropies, 1).mean(1)
        goal_rows = target.expand(n, -1)
        goal_cost_fn = self.geometric_goal_cost(self.model.low_goal_value)
        goal_cost = (
            goal_cost_fn(states[:, -1], goal_rows)
            if goal_cost_fn is not None else latent_l1(states[:, -1], goal_rows)
        )
        cost = goal_cost + self.args.token_prior_weight * prior_nll
        selected = int(cost.argmin())
        return CEMResult(
            token_ids[selected].clone(), states[selected].clone(),
            float(cost[selected]), {
                "goal_cost": float(goal_cost[selected]),
                "token_prior_nll": float(prior_nll[selected]),
                "token_prior_entropy": float(prior_entropy[selected]),
                "shooting_candidates": float(n),
            },
        )

    def macro_plan(self, level_index, start, target, horizon,
                   state_history, action_history, low_history, base_path=None,
                   goal_states=None):
        level = self.model.levels[level_index]
        bank = self.banks[level_index]
        mode = self.args.support_mode
        projection = None
        if mode == "global_bank":
            projection = bank["actions"]

        def ordinary_rollout(codes):
            return level.predictor.rollout(
                start.expand(len(codes), -1), codes,
                state_history=state_history, action_history=action_history,
            )

        def prior_rollout(noise):
            n = len(noise)
            states = state_history.expand(n, -1, -1)
            actions = action_history.expand(n, -1, -1)
            cur = start.expand(n, -1)
            predictions, codes = [], []
            for step in range(noise.shape[1]):
                mu, logvar = level.action.prior_params(cur)
                code = mu + (0.5 * logvar).exp() * noise[:, step]
                actions = torch.cat([actions, code[:, None]], 1)
                cur = level.predictor(states, actions)[:, -1]
                states = torch.cat([states, cur[:, None]], 1)
                predictions.append(cur)
                codes.append(code)
            return torch.stack(predictions, 1), torch.stack(codes, 1)

        def conditional_bank_rollout(raw):
            n = len(raw)
            states = state_history.expand(n, -1, -1)
            actions = action_history.expand(n, -1, -1)
            cur = start.expand(n, -1)
            predictions, codes = [], []
            k = min(self.args.conditional_bank_k, len(bank["states"]))
            for step in range(raw.shape[1]):
                state_ids = torch.cdist(cur, bank["states"]).topk(
                    k, largest=False
                ).indices
                local = bank["actions"][state_ids]
                action_ids = (
                    raw[:, step, None] - local
                ).square().sum(-1).argmin(-1)
                code = local[torch.arange(n, device=self.device), action_ids]
                actions = torch.cat([actions, code[:, None]], 1)
                cur = level.predictor(states, actions)[:, -1]
                states = torch.cat([states, cur[:, None]], 1)
                predictions.append(cur)
                codes.append(code)
            return torch.stack(predictions, 1), torch.stack(codes, 1)

        if mode == "conditional_prior":
            rollout = prior_rollout
        elif mode == "conditional_bank":
            rollout = conditional_bank_rollout
        else:
            rollout = ordinary_rollout
        real_support = bank["support"]

        def extra_cost(codes, predicted):
            starts = torch.cat([start.expand(len(codes), -1)[:, None], predicted[:, :-1]], 1)
            support_raw = level.support(starts, codes)
            support_cost = F.softplus(-support_raw).mean(1)
            nearest = torch.cdist(
                codes.reshape(-1, codes.shape[-1]), bank["actions"]
            ).amin(-1).reshape(len(codes), -1).mean(1)
            addition = torch.zeros_like(support_cost)
            diagnostics = {
                "support_cost": support_cost.detach(),
                "nearest_bank_distance": nearest.detach(),
            }
            if (
                self.advantage_scope == "macro"
                and level_index == len(self.model.levels) - 1
            ):
                goals = target.expand(len(codes), -1)[:, None].expand_as(starts)
                advantage = self.advantage_head(starts, goals, codes).mean(1)
                addition = addition - self.args.advantage_weight * advantage
                diagnostics["distilled_advantage"] = advantage.detach()
            if mode == "gmm":
                mixture = gaussian_mixture_nll(codes, *self.gmms[level_index])
                addition = addition + self.args.gmm_weight * mixture
                diagnostics["gmm_nll"] = mixture.detach()
            if mode == "support_head":
                addition = addition + self.args.support_weight * support_cost
            if self.ensembles is not None and self.args.epistemic_weight > 0:
                member_predictions = []
                for member in self.ensembles[level_index]:
                    member_prediction = member.rollout(
                        start.expand(len(codes), -1), codes,
                        state_history=state_history,
                        action_history=action_history,
                    )
                    member_predictions.append(F.layer_norm(
                        member_prediction, member_prediction.shape[-1:]
                    ))
                epistemic = torch.stack(member_predictions).var(
                    0, unbiased=False
                ).mean((1, 2))
                addition = addition + self.args.epistemic_weight * epistemic
                diagnostics["epistemic_variance"] = epistemic.detach()
            # Percentile: fraction of real chunks with lower support logits.
            percentile = (
                real_support[None] <= support_raw[:, :1]
            ).float().mean(1).squeeze(-1)
            diagnostics["support_percentile"] = percentile.detach()
            return addition, diagnostics

        reachability = None
        if self.args.reachability_refine:
            low_start, low_states, low_actions = low_history

            def reachability(subgoals):
                scale = self.args.reach_budget_scale
                candidates = max(16, int(self.args.token_candidates * scale))
                iterations = max(1, int(self.args.token_iterations * scale))
                return batched_categorical_min_cost(
                    lambda ids: self.low_rollout(
                        low_start, ids, low_states, low_actions
                    ),
                    subgoals,
                    self.model.level_spans[level_index],
                    len(self.vocab),
                    candidates=candidates,
                    iterations=iterations,
                    elites=min(self.args.token_elites, candidates),
                    alpha=self.args.alpha,
                    forbidden=(
                        self.vocab.pad_id,
                        self.vocab.token_to_id[self.vocab.UNK],
                    ),
                    goal_states=(
                        (lambda states: self.lift_low_predictions(
                            base_path, states, level_index
                        )) if self.model.distinct_level_states else None
                    ),
                    rollout_batch_size=self.args.cem_rollout_batch_size,
                )

        init_mean = init_std = None
        if mode == "unconstrained" or mode == "gmm":
            init_mean = bank["actions"].mean(0)
            init_std = bank["actions"].std(0, unbiased=False).clamp_min(0.05)
        result = continuous_cem(
            rollout, target, horizon, level.action.d_macro,
            candidates=self.args.macro_candidates,
            iterations=self.args.macro_iterations,
            elites=self.args.macro_elites, alpha=self.args.alpha,
            init_mean=init_mean, init_std=init_std, project_bank=projection,
            goal_states=goal_states,
            goal_cost_fn=self.geometric_goal_cost(
                self.model.levels[
                    level_index + 1 if goal_states is not None else level_index
                ].goal_value
            ),
            rollout_batch_size=self.args.cem_rollout_batch_size,
            extra_cost=extra_cost,
            reachability=reachability,
            reach_topn=self.args.reach_topn,
            reach_weight=self.args.reach_weight,
        )
        result.diagnostics.update(level=level_index + 1, horizon=horizon)
        return result

    @torch.no_grad()
    def plan_chunk(self, prompt, generated, oracle_goal, oracle_level_goals=None):
        self.oracle_goal = oracle_goal
        prefix = torch.tensor([prompt + generated], device=self.device)
        prefix_states = self.model.encoder(prefix)
        prompt_len = len(prompt)
        start = prefix_states[:, -1]
        low_history = primitive_history(
            self.model, prefix_states, prompt_len, generated
        )
        position = len(generated)

        if self.model.distinct_level_states and not self.args.flat:
            if oracle_level_goals is None:
                raise ValueError("distinct hierarchy requires level-specific goals")
            self.oracle_level_goals = oracle_level_goals
            top_level = len(self.model.levels) - 1
            base_path = prefix_states[:, prompt_len - 1:]
            lifted_paths = self.model.lift_state_path(
                base_path, through_level=top_level
            )
            parent_target = oracle_level_goals[top_level]
            for level_index in reversed(range(len(self.model.levels))):
                span = self.model.level_spans[level_index]
                previous = self.model.level_spans[level_index - 1] if level_index else 1
                ratio = span // previous
                level_path = lifted_paths[level_index]
                boundary_ids = list(range(0, level_path.shape[1], ratio))
                if boundary_ids[-1] != level_path.shape[1] - 1:
                    boundary_ids.append(level_path.shape[1] - 1)
                state_history = level_path[:, boundary_ids]
                level_start = state_history[:, -1]
                complete = len(generated) // span
                if complete:
                    ids = torch.tensor(
                        [generated[:complete * span]], device=self.device
                    )
                    action_history = macro_codes(
                        self.model, ids, through_level=level_index
                    )[level_index]
                else:
                    action_history = level_start.new_zeros(
                        1, 0, self.model.level_dims[level_index]
                    )
                if position % span:
                    state_history = level_start[:, None]
                    action_history = level_start.new_zeros(
                        1, 0, self.model.level_dims[level_index]
                    )
                transform = None
                if level_index < top_level:
                    transform = lambda states, li=level_index, lp=level_path: (
                        self.lift_level_predictions(lp, states, li)
                    )
                horizon = (
                    self.args.high_horizon if level_index == top_level
                    else self.model.level_spans[level_index + 1] // span
                )
                result = self.macro_plan(
                    level_index, level_start, parent_target, horizon,
                    state_history, action_history, (start, *low_history),
                    base_path=base_path, goal_states=transform,
                )
                parent_target = result.states[0:1]
                self.pending.append({
                    "end": position + span, "level": level_index + 1,
                    "predicted": parent_target[0].detach().cpu(),
                    **result.diagnostics,
                })
            token_result = self.token_plan(
                start, parent_target,
                remaining_to_boundary(position, self.model.level_spans[0]),
                *low_history, target_level=0, base_path=base_path,
            )
            execute = (
                min(self.args.token_execution_chunk, len(token_result.actions))
                if self.args.token_execution_chunk > 0
                else len(token_result.actions)
            )
            return token_result.actions[:execute].tolist(), token_result, prefix_states

        if self.args.flat:
            result = self.token_plan(
                start, oracle_goal, self.args.flat_horizon,
                *low_history,
            )
            execute = (
                min(self.args.token_execution_chunk, len(result.actions))
                if self.args.token_execution_chunk > 0
                else min(self.model.level_spans[0], len(result.actions))
            )
            return result.actions[:execute].tolist(), result, prefix_states

        parent_target = oracle_goal
        for level_index in reversed(range(len(self.model.levels))):
            span = self.model.level_spans[level_index]
            if position % span == 0 or level_index not in self.cache:
                if level_index == len(self.model.levels) - 1:
                    horizon = self.args.high_horizon
                else:
                    parent_span = self.model.level_spans[level_index + 1]
                    horizon = (parent_span - position % parent_span) // span
                state_history, action_history = histories(
                    self.model, prefix_states, prompt_len, generated, level_index
                )
                # Feedback MPC can replan a high level between its original
                # fixed boundaries. There is then no completed macro action
                # connecting the old high-level history to the actual current
                # state. Restart the macro predictor from that causal encoder
                # state instead of silently ending history at a stale state.
                if position % span != 0:
                    state_history = start[:, None]
                    action_history = start.new_zeros(
                        1, 0, self.model.level_dims[level_index]
                    )
                    self.records.append({
                        "off_boundary_replan": 1,
                        "off_boundary_level": level_index + 1,
                        "off_boundary_phase": position % span,
                    })
                result = self.macro_plan(
                    level_index, start, parent_target, max(1, horizon),
                    state_history, action_history, (start, *low_history),
                )
                self.cache[level_index] = result.states[0:1]
                self.pending.append({
                    "end": position + span, "level": level_index + 1,
                    "predicted": result.states[0].detach().cpu(),
                    **result.diagnostics,
                })
            parent_target = self.cache[level_index]
        token_horizon = remaining_to_boundary(
            position, self.model.level_spans[0]
        )
        token_result = self.token_plan(
            start, parent_target, token_horizon, *low_history
        )
        execute = (
            min(self.args.token_execution_chunk, len(token_result.actions))
            if self.args.token_execution_chunk > 0
            else len(token_result.actions)
        )
        return token_result.actions[:execute].tolist(), token_result, prefix_states

    @torch.no_grad()
    def observe_chunk(self, prompt, generated_before, tokens, token_result):
        full = torch.tensor([prompt + generated_before + tokens], device=self.device)
        states = self.model.encoder(full)
        target_states = self.model.teacher(full)
        start_index = len(prompt) + len(generated_before)
        actual = target_states[:, start_index:start_index + len(tokens)]
        length = min(actual.shape[1], token_result.states.shape[0])
        drift = latent_l1(
            token_result.states[:length], actual[0, :length]
        )
        teacher_gap = latent_l1(
            states[:, start_index:start_index + length], actual[:, :length]
        )[0]
        if self.oracle_goal is not None:
            self.records.append({
                "actual_goal_distance": float(latent_l1(
                    actual[:, -1], self.oracle_goal
                ))
            })
        for horizon, value in enumerate(drift, 1):
            self.records.append({
                "low_drift_horizon": horizon,
                "low_drift": float(value),
                "online_teacher_gap": float(teacher_gap[horizon - 1]),
            })
        position = len(generated_before) + len(tokens)
        remaining = []
        lower_drift = None
        for pending in self.pending:
            if pending["end"] <= position:
                index = len(prompt) - 1 + pending["end"]
                if index < target_states.shape[1]:
                    actual_macro = target_states[:, index]
                    if self.model.distinct_level_states:
                        level_index = pending["level"] - 1
                        reasoning = target_states[:, len(prompt) - 1:index + 1]
                        actual_macro = self.model.lift_state_path(
                            reasoning, through_level=level_index, teacher=True
                        )[level_index][:, -1]
                    pending["macro_drift"] = float(latent_l1(
                        pending["predicted"].to(self.device)[None], actual_macro,
                    ))
                    if pending["level"] == 1:
                        lower_drift = pending["macro_drift"]
                self.records.append(pending)
            else:
                remaining.append(pending)
        self.pending = remaining
        invalidated = feedback_levels_to_invalidate(
            self.args.feedback_mode,
            lower_drift,
            self.args.feedback_threshold,
            len(self.model.levels),
        )
        for level_index in invalidated:
            self.cache.pop(level_index, None)
        if invalidated:
            kept = []
            for pending in self.pending:
                if pending["level"] - 1 in invalidated:
                    self.records.append({
                        "abandoned_upper_waypoint": 1,
                        "level": pending["level"],
                    })
                else:
                    kept.append(pending)
            self.pending = kept
        self.records.append({
            "feedback_replan": int(bool(invalidated)),
            "feedback_trigger_drift": (
                float(lower_drift) if lower_drift is not None else float("nan")
            ),
        })


def aggregate_records(records):
    keys = sorted({key for row in records for key in row if isinstance(row[key], (int, float))})
    out = {}
    for key in keys:
        values = [float(row[key]) for row in records if key in row and np.isfinite(row[key])]
        if values:
            out[key] = {"mean": float(np.mean(values)), "p90": float(np.quantile(values, .9)), "n": len(values)}
    return out


def grouped_drift(records, group_key, value_key):
    groups = {}
    for row in records:
        if group_key in row and value_key in row and np.isfinite(row[value_key]):
            groups.setdefault(int(row[group_key]), []).append(float(row[value_key]))
    return {
        str(group): {
            "mean": float(np.mean(values)),
            "p90": float(np.quantile(values, .9)),
            "n": len(values),
        }
        for group, values in sorted(groups.items())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--support-mode", choices=[
        "unconstrained", "global_bank", "conditional_bank",
        "gmm", "conditional_prior", "support_head",
    ], default="unconstrained")
    parser.add_argument("--reachability-refine", action="store_true")
    parser.add_argument("--flat", action="store_true")
    parser.add_argument("--feedback-mode", choices=[
        "boundary", "l1_feedback", "adaptive",
    ], default="boundary")
    parser.add_argument("--feedback-threshold", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--high-horizon", type=int, default=2)
    parser.add_argument("--flat-horizon", type=int, default=32)
    parser.add_argument("--macro-candidates", type=int, default=256)
    parser.add_argument("--macro-iterations", type=int, default=5)
    parser.add_argument("--macro-elites", type=int, default=32)
    parser.add_argument("--token-candidates", type=int, default=256)
    parser.add_argument("--token-iterations", type=int, default=5)
    parser.add_argument("--token-elites", type=int, default=32)
    parser.add_argument(
        "--cem-rollout-batch-size", type=int, default=0,
        help="microbatch model rollouts without changing the CEM population",
    )
    parser.add_argument("--token-proposal", choices=[
        "uniform", "prior_energy", "prior_shooting", "prior_greedy",
    ], default="uniform")
    parser.add_argument("--token-prior-weight", type=float, default=0.0)
    parser.add_argument("--token-prior-temperature", type=float, default=1.0)
    parser.add_argument("--token-prior-topk", type=int, default=0)
    parser.add_argument(
        "--token-execution-chunk", type=int, default=0,
        help="execute this many planned tokens before replanning; 0 executes the span",
    )
    parser.add_argument("--reach-topn", type=int, default=16)
    parser.add_argument("--reach-weight", type=float, default=1.0)
    parser.add_argument("--reach-budget-scale", type=float, default=0.4)
    parser.add_argument("--support-weight", type=float, default=0.1)
    parser.add_argument("--gmm-weight", type=float, default=0.1)
    parser.add_argument("--conditional-bank-k", type=int, default=256)
    parser.add_argument("--bank-examples", type=int, default=256)
    parser.add_argument("--bank-size", type=int, default=2048)
    parser.add_argument("--gmm-components", type=int, default=8)
    parser.add_argument("--ensemble-path", default=None)
    parser.add_argument("--epistemic-weight", type=float, default=0.0)
    parser.add_argument("--advantage-head", default=None)
    parser.add_argument("--advantage-weight", type=float, default=1.0)
    parser.add_argument("--goal-score", choices=[
        "latent_distance", "learned_value", "combined",
    ], default="latent_distance")
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--bank-cache", default=None)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--out", default=None)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model, vocab, cfg = load_model(args.ckpt, args.device)
    cache = args.bank_cache
    if cache is None:
        cache = str(Path(args.ckpt).parent / (
            f"macro_bank_e{args.bank_examples}_n{args.bank_size}.pt"
        ))
    banks = load_or_build_banks(
        model, cfg, vocab, args.device, args.bank_examples, args.bank_size, cache
    )
    gmms = (
        [fit_gmm(bank["actions"], args.gmm_components) for bank in banks]
        if args.support_mode == "gmm" else [None] * len(banks)
    )
    ensembles = None
    if args.ensemble_path:
        ensemble_payload = torch.load(
            args.ensemble_path, map_location="cpu", weights_only=False
        )
        ensembles = []
        if len(ensemble_payload["levels"]) != len(model.levels):
            raise ValueError("ensemble hierarchy does not match checkpoint")
        for level, saved_members in zip(model.levels, ensemble_payload["levels"]):
            members = []
            for state in saved_members:
                member = copy.deepcopy(level.predictor)
                member.load_state_dict(state)
                member.to(args.device).eval()
                members.append(member)
            ensembles.append(members)
    dataset = make_dataset(cfg, vocab, args.episodes, cfg.data.val_seed)
    totals = {"success": 0, "valid": 0, "invalid": 0, "tokens": 0}
    all_records = []
    for episode in range(args.episodes):
        item = dataset[episode]
        problem, _ = dataset.igsm.problem(episode)
        prompt = item["tokens"][:item["prompt_len"]]
        generated = []
        full = torch.tensor([item["tokens"]], device=args.device)
        with torch.no_grad():
            teacher_path = model.teacher(full)
            oracle_goal = teacher_path[:, -1]
            reasoning_path = teacher_path[:, item["prompt_len"] - 1:]
            oracle_level_goals = (
                tuple(path[:, -1] for path in model.lift_state_path(
                    reasoning_path, teacher=True
                )) if model.distinct_level_states else None
            )
        planner = OracleCEMPlanner(model, vocab, banks, gmms, args, ensembles)
        while len(generated) < args.max_tokens:
            chunk, result, _ = planner.plan_chunk(
                prompt, generated, oracle_goal, oracle_level_goals
            )
            before = list(generated)
            generated.extend(chunk)
            planner.observe_chunk(prompt, before, chunk, result)
            solved, valid, invalid = replay(problem, vocab, generated)
            if solved:
                break
        solved, valid, invalid = replay(problem, vocab, generated)
        totals["success"] += int(solved)
        totals["valid"] += valid
        totals["invalid"] += invalid
        totals["tokens"] += len(generated)
        all_records.extend(planner.records)
    result = {
        "support_mode": args.support_mode,
        "reachability_refine": args.reachability_refine,
        "flat": args.flat,
        "feedback_mode": args.feedback_mode,
        "oracle_goal": True,
        "goal_score": args.goal_score,
        "value_weight": args.value_weight,
        "uses_auxiliary_lm": False,
        "token_proposal": args.token_proposal,
        "success": totals["success"] / args.episodes,
        "valid_sentences_per_episode": totals["valid"] / args.episodes,
        "invalid_sentences_per_episode": totals["invalid"] / args.episodes,
        "tokens_per_episode": totals["tokens"] / args.episodes,
        "diagnostics": aggregate_records(all_records),
        "execution_drift_by_token_horizon": grouped_drift(
            all_records, "low_drift_horizon", "low_drift"
        ),
        "macro_drift_by_level": grouped_drift(
            all_records, "level", "macro_drift"
        ),
        "args": vars(args),
    }
    tag = f"_{args.output_tag}" if args.output_tag else ""
    dest = Path(args.out) if args.out else Path(args.ckpt).parent / (
        f"oracle_cem_{'flat' if args.flat else args.support_mode}"
        f"_reach{int(args.reachability_refine)}{tag}.json"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
