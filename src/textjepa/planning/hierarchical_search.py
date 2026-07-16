"""Deployable top-down planning with learned macro actions.

The high level samples or optimizes macro codes without enumerating future
environment actions. Its first predicted waypoint becomes a subgoal for a
low-level selector that ranks only the actions feasible in the current state.
"""

from __future__ import annotations

import random
import math

import torch
import torch.nn.functional as F

from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.graph import Problem
from textjepa.data.igsm.render import prompt_sentences
from textjepa.planning.search import EpisodeResult, LatentPlanner, _sequences


class HierarchicalLatentPlanner(LatentPlanner):
    def __init__(
        self,
        *args,
        method: str = "shooting",
        high_horizon: int = 2,
        n_samples: int = 1200,
        cem_iters: int = 20,
        n_elites: int = 20,
        elite_frac: float = 0.1,
        mean_ema: float = 0.0,
        variance_ema: float = 0.9,
        scale_update: str = "variance",
        cem_return: str = "mean",
        cem_tolerance: float = 0.0,
        min_std: float = 0.05,
        cem_domain: str = "code",
        density_weight: float = 0.1,
        macro_q_aux_weight: float = 0.0,
        path_value_weight: float = 0.0,
        learned_support_weight: float = 0.0,
        learned_support_threshold: float | None = None,
        macro_knn_weight: float = 0.0,
        macro_gmm_weight: float = 0.0,
        macro_gmm_components: int = 4,
        macro_gmm_ridge: float = 0.05,
        macro_project_to_span: bool = False,
        reachability_weight: float = 0.0,
        reachability_mode: str = "joint",
        reachability_topk: int = 32,
        measured_reachability_weight: float = 0.0,
        measured_reachability_topk: int = 8,
        measured_reachability_horizon: int = 0,
        measured_latent_goal_weight: float = 0.0,
        measured_symbolic_goal_weight: float = 0.0,
        controller_remaining_weight: float = 0.0,
        controller_residual_weight: float = 0.0,
        collect_controller_outcomes: bool = False,
        ensemble_models: list | None = None,
        epistemic_weight: float = 0.0,
        low_subgoal_weight: float = 1.0,
        low_value_weight: float = 0.0,
        low_method: str = "discrete",
        low_action_source: str = "current",
        low_horizon: int = 1,
        low_max_expand: int = 128,
        low_cem_samples: int = 1200,
        low_cem_iters: int = 5,
        low_cem_elites: int = 20,
        low_cem_variance_ema: float = 0.8,
        low_cem_min_std: float = 0.05,
        low_density_weight: float = 0.1,
        low_support_weight: float = 0.0,
        low_support_threshold: float | None = None,
        low_cem_return: str = "mean",
        allow_oracle_low_actions: bool = False,
        subgoal_source: str = "model",
        discrete_execute_macro: bool = False,
        discrete_first_value_weight: float = 0.0,
        flat_fallback_threshold: float | None = None,
        adaptive_high_horizon: bool = False,
        **kwargs,
    ):
        super().__init__(*args, lookahead=1, hierarchy=True, **kwargs)
        if method not in {"shooting", "cem"}:
            raise ValueError(f"unknown hierarchical planning method: {method}")
        if scale_update not in {"variance", "std"}:
            raise ValueError(f"unknown CEM scale update: {scale_update}")
        if cem_return not in {"mean", "best"}:
            raise ValueError(f"unknown CEM return mode: {cem_return}")
        if cem_domain not in {"code", "prior_noise"}:
            raise ValueError(f"unknown CEM optimization domain: {cem_domain}")
        if reachability_mode not in {"joint", "rerank"}:
            raise ValueError(
                f"unknown reachability integration: {reachability_mode}"
            )
        if low_method not in {"discrete", "cem", "goal_policy"}:
            raise ValueError(f"unknown low-level method: {low_method}")
        if low_action_source not in {"current", "all_problem", "oracle_feasible"}:
            raise ValueError(
                f"unknown low-level action source: {low_action_source}"
            )
        if low_cem_return not in {"mean", "best"}:
            raise ValueError(f"unknown low-level CEM return: {low_cem_return}")
        if subgoal_source not in {
            "model", "oracle_waypoint", "discrete_model", "discrete_true",
            "discrete_all", "discrete_support",
        }:
            raise ValueError(f"unknown subgoal source: {subgoal_source}")
        if (
            low_method == "discrete"
            and low_horizon > 1
            and low_action_source == "oracle_feasible"
            and not allow_oracle_low_actions
        ):
            raise ValueError(
                "low_horizon > 1 enumerates future feasible actions using "
                "the reference graph; set allow_oracle_low_actions=true only "
                "for a labeled hierarchy diagnostic"
            )
        self.method = method
        self.high_horizon = high_horizon
        self.n_samples = n_samples
        self.cem_iters = cem_iters
        self.n_elites = n_elites
        self.elite_frac = elite_frac
        self.mean_ema = mean_ema
        self.variance_ema = variance_ema
        self.scale_update = scale_update
        self.cem_return = cem_return
        self.cem_tolerance = cem_tolerance
        self.min_std = min_std
        self.cem_domain = cem_domain
        self.density_weight = density_weight
        self.macro_q_aux_weight = macro_q_aux_weight
        self.path_value_weight = path_value_weight
        self.learned_support_weight = learned_support_weight
        self.learned_support_threshold = learned_support_threshold
        self.macro_knn_weight = macro_knn_weight
        self.macro_gmm_weight = macro_gmm_weight
        self.macro_gmm_components = macro_gmm_components
        self.macro_gmm_ridge = macro_gmm_ridge
        self.macro_project_to_span = macro_project_to_span
        self.reachability_weight = reachability_weight
        self.reachability_mode = reachability_mode
        self.reachability_topk = reachability_topk
        self.measured_reachability_weight = measured_reachability_weight
        self.measured_reachability_topk = measured_reachability_topk
        self.measured_reachability_horizon = measured_reachability_horizon
        self.measured_latent_goal_weight = measured_latent_goal_weight
        self.measured_symbolic_goal_weight = measured_symbolic_goal_weight
        self.controller_remaining_weight = controller_remaining_weight
        self.controller_residual_weight = controller_residual_weight
        self.collect_controller_outcomes = collect_controller_outcomes
        self.ensemble_models = ensemble_models or []
        self.epistemic_weight = epistemic_weight
        self.low_subgoal_weight = low_subgoal_weight
        self.low_value_weight = low_value_weight
        self.low_method = low_method
        self.low_action_source = low_action_source
        self.low_horizon = low_horizon
        self.low_max_expand = low_max_expand
        self.low_cem_samples = low_cem_samples
        self.low_cem_iters = low_cem_iters
        self.low_cem_elites = low_cem_elites
        self.low_cem_variance_ema = low_cem_variance_ema
        self.low_cem_min_std = low_cem_min_std
        self.low_density_weight = low_density_weight
        self.low_support_weight = low_support_weight
        self.low_support_threshold = low_support_threshold
        self.low_cem_return = low_cem_return
        self.allow_oracle_low_actions = allow_oracle_low_actions
        self.subgoal_source = subgoal_source
        self.discrete_execute_macro = discrete_execute_macro
        self.discrete_first_value_weight = discrete_first_value_weight
        self.flat_fallback_threshold = flat_fallback_threshold
        self.adaptive_high_horizon = adaptive_high_horizon
        self.cem_traces: list[list[dict[str, float]]] = []
        self.low_cem_traces: list[list[dict[str, float]]] = []
        self._macro_reference_codes: torch.Tensor | None = None
        self._macro_gmm: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor
        ] | None = None
        self._low_reachable_states: torch.Tensor | None = None
        self.n_macro_decisions = 0
        self.n_flat_decisions = 0
        self.high_horizon_counts: dict[int, int] = {}
        self._sequence_rng = random.Random(0)
        self.discrete_plan_diagnostics: list[dict[str, float]] = []
        self.measured_reachability_diagnostics: list[dict[str, float]] = []
        self.controller_outcome_batches: list[dict[str, torch.Tensor]] = []

    @staticmethod
    def _ln_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (F.layer_norm(x, x.shape[-1:]) - F.layer_norm(
            y, y.shape[-1:]
        )).abs().mean(-1)

    @staticmethod
    def _is_faithful(problem) -> bool:
        return hasattr(problem, "params") and hasattr(problem, "necessary")

    def _environment(self, problem):
        if self._is_faithful(problem):
            from textjepa.data.faithful import FaithfulEnv

            return FaithfulEnv(problem)
        return SymbolicEnv(problem)

    def _necessary_actions(self, problem) -> set:
        return (
            set(problem.necessary)
            if self._is_faithful(problem)
            else set(problem.query_ancestors)
        )

    def _problem_actions(self, problem) -> list:
        if self._is_faithful(problem):
            return list(problem.action_order)
        return [variable.idx for variable in problem.vars]

    def _prompt_sentences(self, problem, seed: int) -> list[str]:
        if self._is_faithful(problem):
            return list(problem.prompt_sentences)
        return prompt_sentences(problem, random.Random(seed))

    def _action_codes(self, problem, actions: list) -> torch.Tensor:
        if self._is_faithful(problem):
            texts = [f"Define {problem.names[action]} ." for action in actions]
        else:
            from textjepa.data.igsm.render import action_phrase

            texts = [action_phrase(problem, action) for action in actions]
        tokens = self._tokens(texts).squeeze(0).unsqueeze(1)
        return self.model.encode_actions(tokens).squeeze(1)

    @torch.no_grad()
    def _oracle_goal_state(
        self, problem, prompt_tokens, prompt_mask
    ) -> torch.Tensor:
        env = self._environment(problem)
        necessary = self._necessary_actions(problem)
        texts = []
        while not env.solved:
            choices = [
                action for action in env.feasible_actions()
                if action in necessary
            ]
            texts.append(env.step(sorted(choices, key=str)[0]))
        return self._encode_steps(prompt_tokens, prompt_mask, texts)

    def _high_cost(
        self,
        start: torch.Tensor,
        codes: torch.Tensor,
        predictions: torch.Tensor,
        final: torch.Tensor,
        s0: torch.Tensor,
        support: torch.Tensor,
        goal_state: torch.Tensor | None,
    ) -> torch.Tensor:
        n = final.shape[0]
        if self.energy == "oracle_goal":
            if goal_state is None:
                raise ValueError("oracle_goal energy requires a goal state")
            goal = goal_state.expand(final.shape[0], -1)
            task = self._ln_l1(final, goal)
        elif self.energy == "macro_q":
            # Q(s, m) is trained as exact remaining necessary steps after a
            # macro.  At planning time the first macro is the only decision
            # that will be executed before replanning, so score that decision
            # at the observed (on-manifold) current state.
            task = self.model.core.macro_value_head(
                start.expand(n, -1), s0.expand(n, -1), codes[:, 0]
            )
        else:
            task = self.model.core.hi_value_head(
                final, s0.expand(final.shape[0], -1)
            )
        cost = task + self.density_weight * support / self.high_horizon
        if self.macro_q_aux_weight:
            first_q = self.model.core.macro_value_head(
                start.expand(n, -1), s0.expand(n, -1), codes[:, 0]
            )
            cost = cost + self.macro_q_aux_weight * first_q
        if self.path_value_weight:
            path_value = self.model.core.hi_value_head(
                predictions,
                s0.expand(n, -1).unsqueeze(1).expand_as(predictions),
            ).mean(-1)
            cost = cost + self.path_value_weight * path_value
        if self.macro_knn_weight and self._macro_reference_codes is not None:
            # Text macro actions occupy a sparse, multimodal set.  This exact
            # phrase-span manifold term uses no future feasibility labels: it
            # only asks whether the first code is near some K-tuple built from
            # the action phrases available in the current problem.
            distance = torch.cdist(
                codes[:, 0], self._macro_reference_codes
            ).square().min(-1).values / codes.shape[-1]
            cost = cost + self.macro_knn_weight * distance
        if self.macro_gmm_weight and self._macro_gmm is not None:
            cost = cost + self.macro_gmm_weight * self._macro_gmm_nll(
                codes[:, 0]
            )
        if (
            self.reachability_mode == "joint"
            and self.reachability_weight
            and self._low_reachable_states is not None
        ):
            # In this discrete playground we can cheaply approximate the HWM
            # low-level reachability residual with an exhaustive bank of
            # endpoints predicted from deployable K-step action sequences.
            # This feeds the lower-level interface back into every high-level
            # candidate instead of accepting an unreachable first subgoal.
            cost = cost + self.reachability_weight * (
                self._reachability_residual(predictions)
            )
        if self.epistemic_weight and len(self.ensemble_models) > 1:
            members = torch.stack([
                self._predict_macro_states_with(
                    member, start, codes
                )
                for member in self.ensemble_models
            ])
            disagreement = members.var(0, unbiased=False).mean(dim=(1, 2))
            cost = cost + self.epistemic_weight * disagreement
        if (
            self.learned_support_weight
            or self.learned_support_threshold is not None
        ):
            previous = torch.cat(
                [start.expand(n, -1).unsqueeze(1), predictions[:, :-1]], dim=1
            )
            logits = self.model.core.macro_support_head(previous, codes)
            # softplus(-logit) is the positive-class logistic loss: it is low
            # only where the learned conditional support regards (s, m) as a
            # valid macro action.
            if self.learned_support_weight:
                support_cost = F.softplus(-logits).mean(-1)
                cost = cost + self.learned_support_weight * support_cost
            if self.learned_support_threshold is not None:
                invalid = (logits < self.learned_support_threshold).any(-1)
                cost = cost + invalid.float() * 1e4
        return cost

    def _reachability_residual(
        self, predictions: torch.Tensor
    ) -> torch.Tensor:
        return torch.cdist(
            predictions[:, 0], self._low_reachable_states
        ).square().min(-1).values / predictions.shape[-1]

    def _roll_macro_sequences(
        self, start: torch.Tensor, codes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n, horizon, _ = codes.shape
        predictions = self._predict_macro_states(start, codes)
        support = torch.zeros(n, device=start.device)
        macro = self.model.core.macro_encoder
        for h in range(horizon):
            cur = (
                start.expand(n, -1)
                if h == 0 else predictions[:, h - 1]
            )
            pm, pl = macro.prior_params(cur)
            code = codes[:, h]
            support = support + 0.5 * (
                pl + (code - pm).square() * (-pl).exp()
            ).sum(-1)
        return predictions[:, 0], predictions[:, -1], support, predictions

    def _predict_macro_states(
        self, start: torch.Tensor, codes: torch.Tensor
    ) -> torch.Tensor:
        return self._predict_macro_states_with(self.model, start, codes)

    @staticmethod
    def _predict_macro_states_with(
        model, start: torch.Tensor, codes: torch.Tensor
    ) -> torch.Tensor:
        high = model.core.hi_predictor
        if hasattr(high, "rollout"):
            return high.rollout(start, codes)
        cur = start.expand(codes.shape[0], -1)
        result = []
        for h in range(codes.shape[1]):
            cur = high(cur, codes[:, h])
            result.append(cur)
        return torch.stack(result, 1)

    def _shooting_codes(self, start: torch.Tensor) -> torch.Tensor:
        cur = start.expand(self.n_samples, -1)
        codes = []
        macro = self.model.core.macro_encoder
        for _ in range(self.high_horizon):
            code = macro.sample_prior(cur)
            codes.append(code)
            stacked = torch.stack(codes, 1)
            cur = self._predict_macro_states(start, stacked)[:, -1]
        return torch.stack(codes, 1)

    def _codes_from_prior_noise(
        self, start: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Map base Gaussian noise through the learned conditional prior.

        Optimizing ``noise`` rather than arbitrary macro vectors keeps each
        candidate in the image of ``p(u_h | z_h)`` while retaining a smooth
        CEM search space.
        """
        macro = self.model.core.macro_encoder
        codes: list[torch.Tensor] = []
        cur = start.expand(noise.shape[0], -1)
        for step in range(self.high_horizon):
            mean, logvar = macro.prior_params(cur)
            code = mean + (0.5 * logvar).exp() * noise[:, step]
            codes.append(code)
            stacked = torch.stack(codes, 1)
            cur = self._predict_macro_states(start, stacked)[:, -1]
        return torch.stack(codes, 1)

    def _cem_codes(
        self,
        start: torch.Tensor,
        s0: torch.Tensor,
        goal_state: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.cem_domain == "prior_noise":
            mean = start.new_zeros(
                self.high_horizon, self.model.core.macro_encoder.d_macro
            )
            std = torch.ones_like(mean)
        else:
            macro = self.model.core.macro_encoder
            means, stds = [], []
            cur = start
            for _ in range(self.high_horizon):
                mu, lv = macro.prior_params(cur)
                means.append(mu.squeeze(0))
                stds.append((0.5 * lv).exp().squeeze(0))
                stacked = torch.stack(means).unsqueeze(0)
                cur = self._predict_macro_states(start, stacked)[:, -1]
            mean, std = torch.stack(means), torch.stack(stds)
        n_elite = (
            self.n_elites
            if self.n_elites > 0
            else max(2, int(self.n_samples * self.elite_frac))
        )
        n_elite = min(n_elite, self.n_samples)
        best_code = mean.clone()
        best_cost = float("inf")
        trace: list[dict[str, float]] = []
        last_samples: torch.Tensor | None = None
        last_cost: torch.Tensor | None = None
        for iteration in range(self.cem_iters):
            search_samples = mean.unsqueeze(0) + torch.randn(
                self.n_samples, *mean.shape, device=start.device
            ) * std.unsqueeze(0)
            samples = (
                self._codes_from_prior_noise(start, search_samples)
                if self.cem_domain == "prior_noise"
                else search_samples
            )
            _, final, support, predictions = self._roll_macro_sequences(
                start, samples
            )
            cost = self._high_cost(
                start, samples, predictions, final, s0, support, goal_state
            )
            last_samples, last_cost = samples, cost
            sample_best, sample_idx = cost.min(0)
            if float(sample_best) < best_cost:
                best_cost = float(sample_best)
                best_code = samples[int(sample_idx)].clone()
            elite = search_samples[cost.topk(n_elite, largest=False).indices]
            new_mean = elite.mean(0)
            new_var = elite.var(0, unbiased=False)
            mean = self.mean_ema * mean + (1.0 - self.mean_ema) * new_mean
            if self.scale_update == "variance":
                # HWM smooths variance updates to avoid premature collapse.
                var = self.variance_ema * std.square() + (
                    1.0 - self.variance_ema
                ) * new_var
                std = var.sqrt()
            else:
                # Classical continuous CEM smooths the Gaussian scale itself.
                new_std = new_var.sqrt()
                std = self.variance_ema * std + (
                    1.0 - self.variance_ema
                ) * new_std
            std = std.clamp_min(self.min_std)
            trace.append({
                "iteration": float(iteration),
                "sample_best": float(sample_best),
                "best_so_far": best_cost,
                "sample_mean": float(cost.mean()),
                "elite_mean": float(cost.topk(n_elite, largest=False).values.mean()),
                "std_mean": float(std.mean()),
                "std_max": float(std.max()),
            })
            if self.cem_tolerance and float(std.max()) <= self.cem_tolerance:
                break
        self.cem_traces.append(trace)
        if (
            (
                self.measured_reachability_weight
                or self.measured_latent_goal_weight
                or self.measured_symbolic_goal_weight
                or self.controller_remaining_weight
                or self.controller_residual_weight
                or self.collect_controller_outcomes
            )
            and last_samples is not None
            and last_cost is not None
        ):
            # Preserve a small set of the best model-scored plans for the
            # expensive counterfactual closed-loop executability diagnostic.
            # It is reranked later, where the environment and text prefix are
            # available; no symbolic quality labels enter the CEM objective.
            topk = min(self.measured_reachability_topk, len(last_samples))
            return last_samples[
                last_cost.topk(topk, largest=False).indices
            ]
        if (
            self.reachability_mode == "rerank"
            and self.reachability_weight
            and self._low_reachable_states is not None
            and last_samples is not None
            and last_cost is not None
        ):
            topk = min(self.reachability_topk, last_samples.shape[0])
            candidates = last_samples[
                last_cost.topk(topk, largest=False).indices
            ]
            _, final, support, predictions = self._roll_macro_sequences(
                start, candidates
            )
            rerank = self._high_cost(
                start, candidates, predictions, final, s0, support,
                goal_state,
            ) + self.reachability_weight * self._reachability_residual(
                predictions
            )
            return candidates[int(rerank.argmin())].unsqueeze(0)
        if self.cem_return == "best":
            return best_code.unsqueeze(0)
        if self.cem_domain == "prior_noise":
            return self._codes_from_prior_noise(start, mean.unsqueeze(0))
        return mean.unsqueeze(0)

    def _high_subgoal(
        self,
        state: torch.Tensor,
        s0: torch.Tensor,
        goal_state: torch.Tensor | None,
        problem: Problem | None = None,
        env: SymbolicEnv | None = None,
        prompt_tokens: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        step_texts: list[str] | None = None,
    ) -> torch.Tensor:
        if self.method == "shooting":
            codes = self._shooting_codes(state)
        else:
            codes = self._cem_codes(state, s0, goal_state)
        if self.macro_project_to_span and self._macro_reference_codes is not None:
            nearest = torch.cdist(
                codes[:, 0], self._macro_reference_codes
            ).argmin(-1)
            codes[:, 0] = self._macro_reference_codes[nearest]
        first, final, support, predictions = self._roll_macro_sequences(
            state, codes
        )
        cost = self._high_cost(
            state, codes, predictions, final, s0, support, goal_state
        )
        base_cost = cost.clone()
        base_chosen = int(cost.argmin())
        expanded_state = state.expand(len(codes), -1)
        expanded_initial = s0.expand(len(codes), -1)
        if self.controller_remaining_weight:
            predicted_remaining = self.model.core.controller_remaining_head(
                expanded_state, expanded_initial, first
            )
            cost = cost + (
                self.controller_remaining_weight * predicted_remaining
            )
        if self.controller_residual_weight:
            predicted_residual = self.model.core.controller_residual_head(
                expanded_state, expanded_initial, first
            )
            cost = cost + (
                self.controller_residual_weight * predicted_residual
            )
        measured = None
        if (
            self.measured_reachability_weight
            or self.measured_latent_goal_weight
            or self.measured_symbolic_goal_weight
            or self.collect_controller_outcomes
        ):
            if any(value is None for value in (
                problem, env, prompt_tokens, prompt_mask, step_texts
            )):
                raise ValueError(
                    "measured reachability requires episode context"
                )
            measured, endpoints, remaining, distractors = (
                self._measure_closed_loop_reachability(
                problem,
                env,
                state,
                s0,
                first,
                prompt_tokens,
                prompt_mask,
                step_texts,
                )
            )
            cost = cost + self.measured_reachability_weight * measured
            latent_goal = None
            if self.measured_latent_goal_weight:
                if goal_state is None:
                    raise ValueError(
                        "measured latent-goal reranking requires goal state"
                    )
                latent_goal = self._ln_l1(
                    endpoints, goal_state.expand_as(endpoints)
                )
                cost = cost + self.measured_latent_goal_weight * latent_goal
            if self.measured_symbolic_goal_weight:
                cost = cost + self.measured_symbolic_goal_weight * remaining
            if self.collect_controller_outcomes:
                self.controller_outcome_batches.append({
                    "state": expanded_state.detach().cpu(),
                    "initial": expanded_initial.detach().cpu(),
                    "subgoal": first.detach().cpu(),
                    "predicted_final": final.detach().cpu(),
                    "macro": codes[:, 0].detach().cpu(),
                    "base_cost": base_cost.detach().cpu(),
                    "remaining": remaining.detach().cpu(),
                    "residual": measured.detach().cpu(),
                    "distractors": distractors.detach().cpu(),
                    "endpoint": endpoints.detach().cpu(),
                })
        chosen = int(cost.argmin())
        if measured is not None:
            self.measured_reachability_diagnostics.append({
                "base_cost": float(base_cost[base_chosen]),
                "base_residual": float(measured[base_chosen]),
                "selected_base_cost": float(base_cost[chosen]),
                "selected_residual": float(measured[chosen]),
                "base_remaining": float(remaining[base_chosen]),
                "selected_remaining": float(remaining[chosen]),
                "base_distractors": float(distractors[base_chosen]),
                "selected_distractors": float(distractors[chosen]),
                "changed": float(chosen != base_chosen),
                "n_candidates": float(len(codes)),
            })
        self.last_high_plan = {
            "codes": codes[chosen].detach(),
            "first": first[chosen].detach(),
            "final": final[chosen].detach(),
            "support": support[chosen].detach(),
            "learned_support": self.model.core.macro_support_head(
                state, codes[chosen, 0].unsqueeze(0)
            ).squeeze(0).detach(),
            "macro_q": self.model.core.macro_value_head(
                state, s0, codes[chosen, 0].unsqueeze(0)
            ).squeeze(0).detach(),
            "cost": cost[chosen].detach(),
        }
        if self.epistemic_weight and len(self.ensemble_models) > 1:
            member_first = torch.stack([
                self._predict_macro_states_with(
                    member, state, codes[chosen:chosen + 1]
                )[0, 0]
                for member in self.ensemble_models
            ])
            self.last_high_plan["epistemic"] = member_first.var(
                0, unbiased=False
            ).mean().detach()
        return first[chosen].unsqueeze(0)

    def _measure_closed_loop_reachability(
        self,
        problem: Problem,
        env: SymbolicEnv,
        state: torch.Tensor,
        s0: torch.Tensor,
        subgoals: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_mask: torch.Tensor,
        step_texts: list[str],
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Exact controller-specific subgoal residual (diagnostic oracle).

        Each candidate subgoal is held fixed while the deployable low-level
        policy acts for K steps in a cloned environment and re-encodes the
        actual text state after every transition.  This is intentionally an
        expensive upper bound: it tells us whether controller reachability
        can rescue hierarchy before learning an approximation R_psi(z, g).
        """
        horizon = self.measured_reachability_horizon or self.model.core.macro_k
        endpoints = []
        remaining = []
        distractors = []
        for subgoal in subgoals:
            clone = env.clone()
            texts = list(step_texts)
            current = state
            n_distractor = 0
            for _ in range(horizon):
                if clone.solved:
                    break
                feasible = clone.feasible_actions()
                chosen = self._low_action(
                    problem,
                    feasible,
                    current,
                    s0,
                    subgoal.unsqueeze(0),
                    frozenset(clone.resolved_set),
                )
                n_distractor += int(
                chosen not in self._necessary_actions(problem)
                )
                texts.append(clone.step(chosen))
                current = self._current_state(
                    prompt_tokens, prompt_mask, texts
                )
            endpoints.append(current.squeeze(0))
            remaining.append(len(
                self._necessary_actions(problem) - frozenset(clone.resolved_set)
            ))
            distractors.append(n_distractor)
        endpoint_tensor = torch.stack(endpoints)
        return (
            self._ln_l1(endpoint_tensor, subgoals),
            endpoint_tensor,
            subgoals.new_tensor(remaining),
            subgoals.new_tensor(distractors),
        )

    def _oracle_waypoint(
        self,
        problem: Problem,
        env: SymbolicEnv,
        prompt_tokens: torch.Tensor,
        prompt_mask: torch.Tensor,
        step_texts: list[str],
    ) -> torch.Tensor:
        """True next optimal K-step state: manual-subgoal upper bound."""
        clone = env.clone()
        future: list[str] = []
        for _ in range(self.model.core.macro_k):
            if clone.solved:
                break
            necessary = [
                a for a in clone.feasible_actions()
                if a in self._necessary_actions(problem)
            ]
            future.append(clone.step(min(necessary)))
        return self._encode_steps(
            prompt_tokens, prompt_mask, step_texts + future
        )

    def _true_outcome_states(
        self,
        env: SymbolicEnv,
        seqs: list[list[int]],
        step_texts: list[str],
        prompt_tokens: torch.Tensor,
        prompt_mask: torch.Tensor,
        prefix: int | None = None,
    ) -> torch.Tensor:
        outcomes: list[list[str]] = []
        for seq in seqs:
            clone = env.clone()
            chosen = seq if prefix is None else seq[:prefix]
            outcomes.append(step_texts + [clone.step(a) for a in chosen])
        n = len(outcomes)
        chunks = max(len(texts) for texts in outcomes)
        width = max(
            len(self.vocab.encode(text))
            for texts in outcomes for text in texts
        )
        tokens = torch.full(
            (n, chunks, width),
            self.vocab.pad_id,
            dtype=torch.long,
            device=self.device,
        )
        mask = torch.zeros(n, chunks, dtype=torch.bool, device=self.device)
        for i, texts in enumerate(outcomes):
            for j, text in enumerate(texts):
                ids = self.vocab.encode(text)
                tokens[i, j, :len(ids)] = torch.tensor(ids, device=self.device)
                mask[i, j] = True
        _, states = self.model.encode_states(
            prompt_tokens.expand(n, -1, -1),
            prompt_mask.expand(n, -1),
            tokens,
            mask,
        )
        last = mask.sum(1) - 1
        return states[torch.arange(n, device=self.device), last]

    def _discrete_subgoal(
        self,
        problem: Problem,
        env: SymbolicEnv,
        state: torch.Tensor,
        s0: torch.Tensor,
        goal_state: torch.Tensor | None,
        prompt_tokens: torch.Tensor,
        prompt_mask: torch.Tensor,
        step_texts: list[str],
        use_true_outcomes: bool,
        all_problem_actions: bool = False,
        support_guided: bool = False,
    ) -> torch.Tensor:
        """Exhaustive valid macro chunks: on-manifold hierarchy control."""
        K = self.model.core.macro_k
        depth = K * self.high_horizon
        if support_guided:
            seqs = self._support_guided_sequences(
                problem,
                env.feasible_actions(),
                state,
                horizon=depth,
            )
        elif all_problem_actions:
            seqs = self._all_problem_sequences(
                problem,
                env.feasible_actions(),
                frozenset(env.resolved_set),
                horizon=depth,
            )
        else:
            seqs = self._oracle_feasible_sequences(env, depth)
        full = [seq for seq in seqs if len(seq) == depth]
        if not full:
            feasible = env.feasible_actions()
            actions = self._action_codes(problem, feasible)
            predicted = self.model.predictor(
                state.expand(len(feasible), -1), actions
            )
            if goal_state is not None:
                task = self._ln_l1(
                    predicted, goal_state.expand_as(predicted)
                )
            else:
                task = self.model.value_head(
                    predicted, s0.expand_as(predicted)
                )
            chosen = int(task.argmin())
            self.last_discrete_plan = [feasible[chosen]]
            return predicted[chosen].unsqueeze(0)
        if use_true_outcomes:
            first = self._true_outcome_states(
                env, full, step_texts, prompt_tokens, prompt_mask, prefix=K
            )
            final = self._true_outcome_states(
                env, full, step_texts, prompt_tokens, prompt_mask
            )
            support = torch.zeros(len(full), device=self.device)
        else:
            actions = self._action_codes(
                problem, [a for seq in full for a in seq]
            ).reshape(len(full), depth, -1)
            support = torch.zeros(len(full), device=self.device)
            macro_model = self.model.core.macro_encoder
            codes = []
            for h in range(self.high_horizon):
                code = macro_model(
                    actions[:, h * K:(h + 1) * K]
                )
                codes.append(code)
            codes = torch.stack(codes, 1)
            first, final, support, predictions = self._roll_macro_sequences(
                state, codes
            )
        if use_true_outcomes:
            # Discrete true outcomes do not pass through a learned macro code;
            # only state-value/oracle energies are meaningful for this control.
            if self.energy == "macro_q":
                raise ValueError("macro_q cannot score discrete true outcomes")
            if self.energy == "oracle_goal":
                task = self._ln_l1(final, goal_state.expand_as(final))
            else:
                task = self.model.core.hi_value_head(
                    final, s0.expand_as(final)
                )
            cost = task
        else:
            cost = self._high_cost(
                state, codes, predictions, final, s0, support, goal_state
            )
            if self.discrete_first_value_weight:
                first_prediction = self.model.predictor(
                    state.expand(len(full), -1), actions[:, 0]
                )
                first_value = self.model.value_head(
                    first_prediction, s0.expand(len(full), -1)
                )
                cost = cost + self.discrete_first_value_weight * first_value
            if all_problem_actions and (
                self.low_support_weight
                or self.low_support_threshold is not None
            ):
                cur = state.expand(len(full), -1)
                support_cost = torch.zeros(len(full), device=self.device)
                invalid = torch.zeros(
                    len(full), dtype=torch.bool, device=self.device
                )
                for step in range(depth):
                    logits = self.model.core.action_support_head(
                        cur, actions[:, step]
                    )
                    if self.low_support_weight:
                        support_cost += F.softplus(-logits)
                    if self.low_support_threshold is not None:
                        invalid |= logits < self.low_support_threshold
                    cur = self.model.predictor(cur, actions[:, step])
                cost = cost + (
                    self.low_support_weight * support_cost / depth
                )
                if self.low_support_threshold is not None:
                    cost = cost + invalid.float() * 1e4
        chosen = int(cost.argmin())
        self.last_discrete_plan = full[chosen]
        diagnostic_env = env.clone()
        prefix = 0
        before = diagnostic_env.remaining_necessary()
        for action in self.last_discrete_plan:
            if action not in diagnostic_env.feasible_actions():
                break
            diagnostic_env.resolved.append(action)
            prefix += 1
        after = diagnostic_env.remaining_necessary()
        self.discrete_plan_diagnostics.append({
            "valid_prefix": float(prefix),
            "valid_fraction": prefix / len(self.last_discrete_plan),
            "fully_valid": float(prefix == len(self.last_discrete_plan)),
            "oracle_progress": float(before - after),
        })
        return first[chosen].unsqueeze(0)

    def _all_problem_sequences(
        self,
        problem: Problem,
        feasible: list[int],
        resolved: frozenset[int],
        horizon: int | None = None,
    ) -> list[list[int]]:
        """Text-action search without querying future symbolic feasibility."""
        unresolved = [
            action for action in self._problem_actions(problem)
            if action not in resolved
        ]
        frontier = [[action] for action in feasible]
        horizon = self.low_horizon if horizon is None else horizon
        for _ in range(1, horizon):
            expanded = [
                seq + [action]
                for seq in frontier
                for action in unresolved
                if action not in seq
            ]
            if not expanded:
                break
            frontier = self._balanced_first_action_cap(
                expanded, self.low_max_expand, self._sequence_rng
            )
        return frontier

    def _support_guided_sequences(
        self,
        problem: Problem,
        feasible: list,
        state: torch.Tensor,
        horizon: int,
    ) -> list[list]:
        """Build a root-balanced beam using learned future feasibility.

        Current feasibility is observed and therefore fixes the root actions.
        Every later expansion is ranked only by the learned
        ``p(feasible | predicted_state, action)`` head.  Keeping an equal beam
        for each root prevents proposal multiplicity from deciding the action
        that will actually be executed.
        """
        unresolved = self._problem_actions(problem)
        beam_per_root = max(1, self.low_max_expand // max(len(feasible), 1))
        completed: list[list] = []
        for root in feasible:
            root_code = self._action_codes(problem, [root])
            root_state = self.model.predictor(state, root_code)[0]
            frontier: list[tuple[list, torch.Tensor, float]] = [
                ([root], root_state, 0.0)
            ]
            for _ in range(1, horizon):
                parent_states = []
                candidate_actions = []
                parent_rows = []
                for parent, (_, _, _) in enumerate(frontier):
                    sequence = frontier[parent][0]
                    for action in unresolved:
                        if action not in sequence:
                            parent_states.append(frontier[parent][1])
                            candidate_actions.append(action)
                            parent_rows.append(parent)
                if not candidate_actions:
                    frontier = []
                    break
                parent_tensor = torch.stack(parent_states)
                action_tensor = self._action_codes(
                    problem, candidate_actions
                )
                logits = self.model.core.action_support_head(
                    parent_tensor, action_tensor
                )
                next_states = self.model.predictor(
                    parent_tensor, action_tensor
                )
                expanded = []
                for row, (action, parent) in enumerate(
                    zip(candidate_actions, parent_rows)
                ):
                    sequence, _, previous_cost = frontier[parent]
                    cost = previous_cost + float(F.softplus(-logits[row]))
                    expanded.append((
                        sequence + [action], next_states[row], cost
                    ))
                expanded.sort(key=lambda item: item[2])
                frontier = expanded[:beam_per_root]
            completed.extend(
                sequence for sequence, _, _ in frontier
                if len(sequence) == horizon
            )
        return completed[:self.low_max_expand]

    def _oracle_feasible_sequences(
        self, env: SymbolicEnv, horizon: int
    ) -> list[list]:
        """Root-balanced future-feasibility diagnostic.

        This deliberately queries cloned environments after the current step
        and must never be presented as deployable.  It measures whether
        proposal validity, rather than high-level scoring, is the remaining
        bottleneck without reintroducing the lexicographic-cap artifact.
        """
        frontier: list[tuple[list, object]] = [([], env.clone())]
        for _ in range(horizon):
            expanded: list[tuple[list, object]] = []
            for sequence, current in frontier:
                for action in current.feasible_actions():
                    clone = current.clone()
                    clone.step(action)
                    expanded.append((sequence + [action], clone))
            if not expanded:
                break
            if len(expanded) > self.low_max_expand:
                sequences = self._balanced_first_action_cap(
                    [sequence for sequence, _ in expanded],
                    self.low_max_expand,
                    self._sequence_rng,
                )
                by_sequence = {
                    tuple(sequence): clone for sequence, clone in expanded
                }
                frontier = [
                    (sequence, by_sequence[tuple(sequence)])
                    for sequence in sequences
                ]
            else:
                frontier = expanded
        return [
            sequence for sequence, _ in frontier
            if len(sequence) == horizon
        ]

    @staticmethod
    def _balanced_first_action_cap(
        sequences: list[list], cap: int, rng: random.Random | None = None
    ) -> list[list]:
        """Cap a sequence bank without privileging its first action.

        The first action is the decision executed by the receding-horizon
        discrete planner.  A plain lexicographic slice can fill the entire
        bank with descendants of the first feasible action and silently turn
        planning into an ordered-action heuristic.  Round-robin truncation
        gives every current action equal proposal multiplicity up to one.
        """
        if len(sequences) <= cap:
            return sequences
        groups: dict[object, list[list]] = {}
        for sequence in sequences:
            groups.setdefault(sequence[0], []).append(sequence)
        if rng is not None:
            for group in groups.values():
                rng.shuffle(group)
        selected = []
        depth = 0
        ordered_groups = list(groups.values())
        while len(selected) < cap:
            added = False
            for group in ordered_groups:
                if depth < len(group):
                    selected.append(group[depth])
                    added = True
                    if len(selected) == cap:
                        break
            if not added:
                break
            depth += 1
        return selected

    def _prepare_macro_reference_codes(
        self,
        problem: Problem,
        feasible: list[int],
        resolved: frozenset[int],
    ) -> torch.Tensor | None:
        """Encode deployable candidate spans for a text-action manifold."""
        K = self.model.core.macro_k
        seqs = self._all_problem_sequences(
            problem, feasible, resolved, horizon=K
        )
        seqs = [seq for seq in seqs if len(seq) == K]
        if not seqs:
            return None
        actions = self._action_codes(
            problem, [action for seq in seqs for action in seq]
        ).reshape(len(seqs), K, -1)
        return self.model.core.macro_encoder(actions)

    def _fit_macro_gmm(self, codes: torch.Tensor) -> None:
        """Fit a small full-covariance mixture to the local action bank."""
        components = min(self.macro_gmm_components, len(codes))
        if components <= 0:
            self._macro_gmm = None
            return
        indices = torch.linspace(
            0, len(codes) - 1, components, device=codes.device
        ).round().long()
        means = codes[indices].clone()
        for _ in range(8):
            labels = torch.cdist(codes, means).argmin(-1)
            updated = means.clone()
            for component in range(components):
                points = codes[labels == component]
                if len(points):
                    updated[component] = points.mean(0)
            means = updated
        labels = torch.cdist(codes, means).argmin(-1)
        dimension = codes.shape[-1]
        global_scale = codes.var(0, unbiased=False).mean().clamp_min(0.1)
        eye = torch.eye(dimension, device=codes.device, dtype=codes.dtype)
        weights, cholesky = [], []
        for component in range(components):
            points = codes[labels == component]
            if not len(points):
                points = codes
            centered = points - means[component]
            covariance = centered.T @ centered / max(len(points), 1)
            covariance = covariance + (
                self.macro_gmm_ridge * global_scale * eye
            )
            weights.append(codes.new_tensor(len(points) / len(codes)))
            cholesky.append(torch.linalg.cholesky(covariance))
        mixture_weights = torch.stack(weights)
        mixture_weights = mixture_weights / mixture_weights.sum()
        self._macro_gmm = (
            mixture_weights, means, torch.stack(cholesky)
        )

    def _macro_gmm_nll(self, codes: torch.Tensor) -> torch.Tensor:
        weights, means, cholesky = self._macro_gmm
        difference = codes.unsqueeze(1) - means.unsqueeze(0)
        whitened = torch.linalg.solve_triangular(
            cholesky.unsqueeze(0), difference.unsqueeze(-1), upper=False
        ).squeeze(-1)
        mahalanobis = whitened.square().sum(-1)
        logdet = cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        dimension = codes.shape[-1]
        log_prob = (
            weights.log().unsqueeze(0)
            - 0.5 * mahalanobis
            - logdet.unsqueeze(0)
            - 0.5 * dimension * math.log(2.0 * math.pi)
        )
        return -torch.logsumexp(log_prob, -1) / dimension

    def _prepare_low_reachable_states(
        self,
        problem: Problem,
        feasible: list[int],
        resolved: frozenset[int],
        state: torch.Tensor,
    ) -> torch.Tensor | None:
        """Predicted endpoints of problem-local deployable macro spans."""
        K = self.model.core.macro_k
        seqs = self._all_problem_sequences(
            problem, feasible, resolved, horizon=K
        )
        seqs = [seq for seq in seqs if len(seq) == K]
        if not seqs:
            return None
        actions = self._action_codes(
            problem, [action for seq in seqs for action in seq]
        ).reshape(len(seqs), K, -1)
        cur = state.expand(len(seqs), -1)
        for step in range(K):
            cur = self.model.predictor(cur, actions[:, step])
        return cur

    def _low_discrete_action(
        self,
        problem: Problem,
        feasible: list[int],
        state: torch.Tensor,
        s0: torch.Tensor,
        subgoal: torch.Tensor,
        resolved: frozenset[int],
    ) -> int:
        if self.low_horizon == 1 or self.low_action_source == "current":
            seqs = [[a] for a in feasible]
        elif self.low_action_source == "all_problem":
            seqs = self._all_problem_sequences(
                problem, feasible, resolved
            )
        else:
            seqs = _sequences(
                problem, resolved, self.low_horizon, self.low_max_expand
            )
            seqs = [seq for seq in seqs if seq]
        unique = sorted({a for seq in seqs for a in seq})
        encoded = self._action_codes(problem, unique)
        code = {a: encoded[i] for i, a in enumerate(unique)}
        cur = state.expand(len(seqs), -1).clone()
        support_cost = torch.zeros(len(seqs), device=self.device)
        support_invalid = torch.zeros(
            len(seqs), dtype=torch.bool, device=self.device
        )
        max_depth = max(len(seq) for seq in seqs)
        for depth in range(max_depth):
            alive = torch.tensor(
                [depth < len(seq) for seq in seqs],
                dtype=torch.bool,
                device=self.device,
            )
            action = torch.stack([
                code[seq[depth]] if depth < len(seq) else code[seq[0]]
                for seq in seqs
            ])
            if self.low_support_weight or self.low_support_threshold is not None:
                logits = self.model.core.action_support_head(cur, action)
                if self.low_support_weight:
                    support_cost = (
                        support_cost + alive.float() * F.softplus(-logits)
                    )
                if self.low_support_threshold is not None:
                    support_invalid |= alive & (
                        logits < self.low_support_threshold
                    )
            nxt = self.model.predictor(cur, action)
            cur = torch.where(alive.unsqueeze(-1), nxt, cur)
        cost = self.low_subgoal_weight * self._ln_l1(
            cur, subgoal.expand_as(cur)
        )
        if self.low_support_weight:
            lengths = torch.tensor(
                [len(seq) for seq in seqs],
                dtype=cost.dtype,
                device=self.device,
            )
            cost = cost + self.low_support_weight * support_cost / lengths
        if self.low_support_threshold is not None:
            cost = cost + support_invalid.float() * 1e4
        if self.low_value_weight:
            cost = cost + self.low_value_weight * self.model.value_head(
                cur, s0.expand_as(cur)
            )
        return seqs[int(cost.argmin())][0]

    def _low_cem_action(
        self,
        problem: Problem,
        feasible: list[int],
        state: torch.Tensor,
        s0: torch.Tensor,
        subgoal: torch.Tensor,
        resolved: frozenset[int],
    ) -> int:
        """HWM-style low-level CEM for the discrete text-action interface.

        CEM optimizes a continuous sequence of low-level action embeddings.
        A nearest-manifold term keeps every embedding near a real intent
        phrase; only the first optimized embedding is projected onto the set
        of actions currently feasible in the environment.
        """
        unresolved = [
            action for action in self._problem_actions(problem)
            if action not in resolved
        ]
        all_codes = self._action_codes(problem, unresolved)
        feasible_codes = self._action_codes(problem, feasible)
        horizon = self.low_horizon
        all_mean = all_codes.mean(0)
        all_std = all_codes.std(0, unbiased=False).clamp_min(
            self.low_cem_min_std
        )
        mean = all_mean.expand(horizon, -1).clone()
        std = all_std.expand(horizon, -1).clone()
        mean[0] = feasible_codes.mean(0)
        std[0] = feasible_codes.std(0, unbiased=False).clamp_min(
            self.low_cem_min_std
        )
        n_elite = min(self.low_cem_elites, self.low_cem_samples)
        best_code = mean.clone()
        best_cost = float("inf")
        trace: list[dict[str, float]] = []
        for iteration in range(self.low_cem_iters):
            samples = mean.unsqueeze(0) + torch.randn(
                self.low_cem_samples,
                *mean.shape,
                device=self.device,
            ) * std.unsqueeze(0)
            cur = state.expand(self.low_cem_samples, -1)
            learned_support = torch.zeros(
                self.low_cem_samples, device=self.device
            )
            for depth in range(horizon):
                if self.low_support_weight:
                    logits = self.model.core.action_support_head(
                        cur, samples[:, depth]
                    )
                    learned_support = learned_support + F.softplus(-logits)
                cur = self.model.predictor(cur, samples[:, depth])
            cost = self.low_subgoal_weight * self._ln_l1(
                cur, subgoal.expand_as(cur)
            )
            if self.low_value_weight:
                cost = cost + self.low_value_weight * self.model.value_head(
                    cur, s0.expand_as(cur)
                )
            # First action must project to a currently feasible phrase; later
            # actions only need to remain near some unresolved phrase.
            first_distance = torch.cdist(
                samples[:, 0], feasible_codes
            ).square().min(-1).values
            manifold = first_distance
            if horizon > 1:
                future_distance = torch.cdist(
                    samples[:, 1:].reshape(-1, samples.shape[-1]), all_codes
                ).square().min(-1).values.reshape(
                    self.low_cem_samples, horizon - 1
                )
                manifold = manifold + future_distance.mean(-1)
            cost = cost + self.low_density_weight * manifold
            if self.low_support_weight:
                cost = cost + (
                    self.low_support_weight * learned_support / horizon
                )
            sample_best, sample_idx = cost.min(0)
            if float(sample_best) < best_cost:
                best_cost = float(sample_best)
                best_code = samples[int(sample_idx)].clone()
            elite_idx = cost.topk(n_elite, largest=False).indices
            elite = samples[elite_idx]
            new_mean = elite.mean(0)
            new_var = elite.var(0, unbiased=False)
            mean = new_mean
            var = self.low_cem_variance_ema * std.square() + (
                1.0 - self.low_cem_variance_ema
            ) * new_var
            std = var.sqrt().clamp_min(self.low_cem_min_std)
            trace.append({
                "iteration": float(iteration),
                "sample_best": float(sample_best),
                "best_so_far": best_cost,
                "sample_mean": float(cost.mean()),
                "elite_mean": float(cost[elite_idx].mean()),
                "std_mean": float(std.mean()),
                "std_max": float(std.max()),
            })
        self.low_cem_traces.append(trace)
        selected = best_code if self.low_cem_return == "best" else mean
        nearest = torch.cdist(
            selected[0].unsqueeze(0), feasible_codes
        ).argmin()
        return feasible[int(nearest)]

    def _low_action(
        self,
        problem: Problem,
        feasible: list[int],
        state: torch.Tensor,
        s0: torch.Tensor,
        subgoal: torch.Tensor,
        resolved: frozenset[int],
    ) -> int:
        if self.low_method == "goal_policy":
            actions = self._action_codes(problem, feasible)
            cost = self.model.core.subgoal_action_head(
                state.expand(len(feasible), -1),
                subgoal.expand(len(feasible), -1),
                actions,
            )
            return feasible[int(cost.argmin())]
        if self.low_method == "cem":
            return self._low_cem_action(
                problem, feasible, state, s0, subgoal, resolved
            )
        return self._low_discrete_action(
            problem, feasible, state, s0, subgoal, resolved
        )

    def _flat_value_action(
        self,
        problem: Problem,
        feasible: list[int],
        state: torch.Tensor,
        s0: torch.Tensor,
    ) -> int:
        actions = self._action_codes(problem, feasible)
        predicted = self.model.predictor(
            state.expand(len(feasible), -1), actions
        )
        cost = self.model.value_head(
            predicted, s0.expand(len(feasible), -1)
        )
        return feasible[int(cost.argmin())]

    @torch.no_grad()
    def plan_episode(
        self, problem: Problem, slack: int = 0, seed: int = 0
    ) -> EpisodeResult:
        self._sequence_rng = random.Random(seed)
        env = self._environment(problem)
        prompt = self._prompt_sentences(problem, seed)
        prompt_tokens = self._tokens(prompt)
        prompt_mask = torch.ones(
            1, len(prompt), dtype=torch.bool, device=self.device
        )
        step_texts: list[str] = []
        necessary = self._necessary_actions(problem)
        n_necessary = len(necessary)
        budget = n_necessary + slack
        n_distractor = 0
        goal_state = (
            self._oracle_goal_state(problem, prompt_tokens, prompt_mask)
            if (
                self.energy == "oracle_goal"
                or self.measured_latent_goal_weight
            )
            else None
        )
        while not env.solved and len(step_texts) < budget:
            state = self._current_state(prompt_tokens, prompt_mask, step_texts)
            s0 = self._s0(prompt_tokens, prompt_mask)
            if (
                self.flat_fallback_threshold is not None
                and float(self.model.value_head(state, s0))
                <= self.flat_fallback_threshold
            ):
                self.n_flat_decisions += 1
                chosen = self._flat_value_action(
                    problem, env.feasible_actions(), state, s0
                )
                n_distractor += int(chosen not in necessary)
                step_texts.append(env.step(chosen))
                continue
            self.n_macro_decisions += 1
            if (
                self.macro_knn_weight
                or self.macro_gmm_weight
                or self.macro_project_to_span
            ):
                self._macro_reference_codes = self._prepare_macro_reference_codes(
                    problem,
                    env.feasible_actions(),
                    frozenset(env.resolved_set),
                )
            else:
                self._macro_reference_codes = None
            if self.macro_gmm_weight and self._macro_reference_codes is not None:
                self._fit_macro_gmm(self._macro_reference_codes)
            else:
                self._macro_gmm = None
            if self.reachability_weight:
                self._low_reachable_states = self._prepare_low_reachable_states(
                    problem,
                    env.feasible_actions(),
                    frozenset(env.resolved_set),
                    state,
                )
            else:
                self._low_reachable_states = None
            if self.subgoal_source == "oracle_waypoint":
                subgoal = self._oracle_waypoint(
                    problem, env, prompt_tokens, prompt_mask, step_texts
                )
            elif self.subgoal_source in {
                "discrete_model", "discrete_true", "discrete_all",
                "discrete_support",
            }:
                subgoal = self._discrete_subgoal(
                    problem,
                    env,
                    state,
                    s0,
                    goal_state,
                    prompt_tokens,
                    prompt_mask,
                    step_texts,
                    use_true_outcomes=self.subgoal_source == "discrete_true",
                    all_problem_actions=self.subgoal_source in {
                        "discrete_all", "discrete_support"
                    },
                    support_guided=(
                        self.subgoal_source == "discrete_support"
                    ),
                )
            else:
                configured_horizon = self.high_horizon
                if self.adaptive_high_horizon:
                    estimated_remaining = max(
                        0.0, float(self.model.value_head(state, s0))
                    )
                    self.high_horizon = max(1, min(
                        configured_horizon,
                        math.ceil(
                            estimated_remaining / self.model.core.macro_k
                        ),
                    ))
                self.high_horizon_counts[self.high_horizon] = (
                    self.high_horizon_counts.get(self.high_horizon, 0) + 1
                )
                try:
                    subgoal = self._high_subgoal(
                        state,
                        s0,
                        goal_state,
                        problem,
                        env,
                        prompt_tokens,
                        prompt_mask,
                        step_texts,
                    )
                finally:
                    self.high_horizon = configured_horizon
            feasible = env.feasible_actions()
            if (
                self.discrete_execute_macro
                and self.subgoal_source in {
                    "discrete_model", "discrete_all", "discrete_support"
                }
            ):
                chosen = self.last_discrete_plan[0]
            else:
                chosen = self._low_action(
                    problem,
                    feasible,
                    state,
                    s0,
                    subgoal,
                    frozenset(env.resolved_set),
                )
            n_distractor += int(chosen not in necessary)
            step_texts.append(env.step(chosen))
        return EpisodeResult(
            env.solved,
            len(step_texts),
            n_necessary,
            n_distractor,
        )
