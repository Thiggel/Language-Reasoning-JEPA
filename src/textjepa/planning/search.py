"""Closed-loop latent planning over discourse actions.

The planner may query the *interface* of the world — which actions are
feasible (dependency preconditions are stated in the prompt) and their intent
phrases. Lookahead 1 enumerates every currently feasible action and is the
deployable, information-matched protocol. Deeper lookahead additionally uses
the reference dependency graph to enumerate future feasible actions and detect
terminal sequences. It is therefore an explicitly opt-in oracle-action
diagnostic, even when consequences are rolled out with the latent predictor.
The chosen action is executed by the environment, whose outcome sentence is
re-encoded before replanning.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.graph import Problem
from textjepa.data.igsm.render import prompt_sentences
from textjepa.data.vocab import Vocab


def validate_learned_catalogue_checkpoint(cfg) -> None:
    """Reject checkpoints that cannot support the claimed planner protocol."""
    errors = []
    if not bool(cfg.model.get("action_prior", False)):
        errors.append("model.action_prior must be enabled")
    if cfg.model.get("action_support_states", "true") != "all":
        errors.append("model.action_support_states must be 'all'")
    if cfg.model.get("action_prior_states", "true") != "all":
        errors.append("model.action_prior_states must be 'all'")
    if cfg.model.get("action_prior_candidate_scope", "feasible") != "catalogue":
        errors.append("model.action_prior_candidate_scope must be 'catalogue'")
    if not bool(cfg.data.get("all_action_supervision", False)):
        errors.append("data.all_action_supervision must be enabled")
    feasibility = cfg.objective.get("action_feasibility", {})
    if float(feasibility.get("weight", 0.0)) <= 0.0:
        errors.append("objective.action_feasibility.weight must be positive")
    prior = cfg.objective.get("action_prior", {})
    if float(prior.get("weight", 0.0)) <= 0.0:
        errors.append("objective.action_prior.weight must be positive")
    if errors:
        raise ValueError(
            "checkpoint is invalid for learned-catalogue planning: "
            + "; ".join(errors)
        )


@dataclass
class EpisodeResult:
    solved: bool
    steps: int
    n_necessary: int
    n_distractor: int
    n_invalid: int = 0


def _feasible(problem: Problem, resolved: frozenset[int]) -> list[int]:
    return [
        v.idx
        for v in problem.vars
        if v.idx not in resolved and all(p in resolved for p in v.parents)
    ]


def _sequences(
    problem: Problem, resolved: frozenset[int], depth: int, cap: int
) -> list[list[int]]:
    """All feasible action sequences up to ``depth`` (stop early if solved)."""
    seqs: list[list[int]] = []
    frontier: list[tuple[list[int], frozenset[int]]] = [([], resolved)]
    for _ in range(depth):
        nxt = []
        for seq, res in frontier:
            for a in _feasible(problem, res):
                new = (seq + [a], res | {a})
                if problem.query in new[1]:
                    seqs.append(new[0])
                else:
                    nxt.append(new)
        frontier = nxt[:cap]
        if not frontier:
            break
    seqs.extend(seq for seq, _ in frontier)
    return seqs[: cap * 4] if seqs else [[]]


class LatentPlanner:
    def __init__(
        self,
        model,
        vocab: Vocab,
        device: torch.device,
        lookahead: int = 1,
        max_expand: int = 64,
        energy: str = "value",  # "value" | "oracle_goal"
        hierarchy: bool = False,  # score K-step sequences with F_hi jumps
        simulator: str = "latent",  # "latent" (F rollouts) | "symbolic"
        allow_oracle_future_actions: bool = False,
        prior_top_m: int = 0,
        prior_only: bool = False,
        proposal_source: str = "current_feasible",
        proposal_top_m: int = 0,
        proposal_beam_width: int = 1,
        proposal_prior_weight: float = 1.0,
        proposal_support_weight: float = 1.0,
    ):
        if proposal_source not in {"current_feasible", "learned_catalogue"}:
            raise ValueError(f"unknown proposal source: {proposal_source}")
        if (
            lookahead > 1
            and proposal_source == "current_feasible"
            and not allow_oracle_future_actions
        ):
            raise ValueError(
                "lookahead > 1 enumerates future actions with the reference "
                "dependency graph; set allow_oracle_future_actions=true "
                "only for a labeled oracle-action diagnostic"
            )
        self.model = model
        self.vocab = vocab
        self.device = device
        self.lookahead = lookahead
        self.max_expand = max_expand
        self.energy = energy
        self.hierarchy = hierarchy
        self.simulator = simulator
        self.allow_oracle_future_actions = allow_oracle_future_actions
        self.prior_top_m = max(0, int(prior_top_m))
        self.prior_only = bool(prior_only)
        self.proposal_source = proposal_source
        self.proposal_top_m = max(0, int(proposal_top_m))
        self.proposal_beam_width = max(1, int(proposal_beam_width))
        self.proposal_prior_weight = float(proposal_prior_weight)
        self.proposal_support_weight = float(proposal_support_weight)
        if self.proposal_source == "learned_catalogue":
            if self.proposal_top_m < 1:
                raise ValueError(
                    "learned-catalogue planning requires proposal_top_m >= 1"
                )
            if allow_oracle_future_actions:
                raise ValueError(
                    "learned-catalogue planning cannot enable oracle future actions"
                )
            if simulator != "latent":
                raise ValueError(
                    "learned-catalogue planning requires latent simulation"
                )
        if (self.prior_top_m or self.prior_only) and getattr(
            model, "action_prior_head", None
        ) is None:
            raise ValueError("prior planning requires an action-prior checkpoint")
        if self.proposal_source == "learned_catalogue" and getattr(
            model, "action_prior_head", None
        ) is None:
            raise ValueError(
                "learned-catalogue planning requires an action-prior checkpoint"
            )
        self.prior_decisions = 0
        self.prior_necessary_recall_sum = 0.0
        self.proposal_decisions = 0
        self.proposal_root_necessary_recall_sum = 0.0
        self.proposal_root_feasible_recall_sum = 0.0
        self.proposal_root_feasible_precision_sum = 0.0

    def _tokens(self, texts: list[str], min_chunks: int = 0) -> torch.Tensor:
        ids = [self.vocab.encode(t) for t in texts]
        C = max(len(ids), min_chunks, 1)
        L = max((len(i) for i in ids), default=1)
        out = torch.full((1, C, L), self.vocab.pad_id, dtype=torch.long)
        for c, i in enumerate(ids):
            out[0, c, : len(i)] = torch.tensor(i)
        return out.to(self.device)

    @torch.no_grad()
    def plan_episode(self, problem: Problem, slack: int = 0, seed: int = 0) -> EpisodeResult:
        env = SymbolicEnv(problem)
        prompt = prompt_sentences(problem, random.Random(seed))
        prompt_tokens = self._tokens(prompt)
        prompt_mask = torch.ones(1, len(prompt), dtype=torch.bool, device=self.device)
        step_texts: list[str] = []
        action_history: list[int] = []
        budget = problem.n_necessary_steps + slack
        n_distractor = 0
        attempts = 0
        self._episode_catalogue = [variable.idx for variable in problem.vars]
        random.Random(f"{seed}:learned-catalogue").shuffle(
            self._episode_catalogue
        )
        goal_state = (
            self._oracle_goal_state(problem, prompt_tokens, prompt_mask)
            if self.energy == "oracle_goal"
            else None
        )

        while not env.solved and attempts < budget:
            s = self._current_state(prompt_tokens, prompt_mask, step_texts)
            s0 = self._s0(prompt_tokens, prompt_mask)
            state_history, action_codes = self._causal_history(
                prompt_tokens,
                prompt_mask,
                step_texts,
                problem,
                action_history,
            )
            proposal_costs = None
            if self.proposal_source == "learned_catalogue":
                seqs, proposal_costs = self._learned_catalogue_sequences(
                    s,
                    problem,
                    executed=action_history,
                    state_history=state_history,
                    action_history=action_codes,
                )
                # Post-hoc diagnostic only: hidden relevance never enters
                # proposal scores, filtering, or trajectory construction.
                proposed_roots = {sequence[0] for sequence in seqs}
                true_feasible = set(env.feasible_actions())
                feasible_necessary = true_feasible & set(
                    problem.query_ancestors
                )
                self.proposal_decisions += 1
                self.proposal_root_necessary_recall_sum += (
                    len(feasible_necessary & proposed_roots)
                    / max(len(feasible_necessary), 1)
                )
                self.proposal_root_feasible_recall_sum += (
                    len(true_feasible & proposed_roots)
                    / max(len(true_feasible), 1)
                )
                self.proposal_root_feasible_precision_sum += (
                    len(true_feasible & proposed_roots)
                    / max(len(proposed_roots), 1)
                )
            else:
                seqs = _sequences(
                    problem,
                    frozenset(env.resolved_set),
                    self.lookahead,
                    self.max_expand,
                )
                seqs = self._apply_action_prior(s, problem, seqs)
            if self.prior_only and proposal_costs is not None:
                best = seqs[int(proposal_costs.argmin().item())]
            else:
                best = self._best_sequence(
                    s, s0, problem, seqs, goal_state,
                    state_history=state_history,
                    action_history=action_codes,
                    sym_ctx=(env, step_texts, prompt_tokens, prompt_mask),
                )
            chosen = best[0]
            n_distractor += int(chosen not in problem.query_ancestors)
            attempts += 1
            try:
                outcome = env.step(chosen)
            except ValueError:
                return EpisodeResult(
                    False,
                    attempts,
                    problem.n_necessary_steps,
                    n_distractor,
                    n_invalid=1,
                )
            step_texts.append(outcome)
            action_history.append(chosen)

        return EpisodeResult(
            env.solved, attempts, problem.n_necessary_steps, n_distractor
        )

    def _proposal_scores(
        self,
        state: torch.Tensor,
        action_codes: torch.Tensor,
        action_history: torch.Tensor | None = None,
        support_codes: torch.Tensor | None = None,
        support_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Log-score catalogue actions using only learned model heads."""
        n = action_codes.shape[0]
        expanded = state.expand(n, -1)
        support_codes = action_codes if support_codes is None else support_codes
        history = history_mask = None
        history_source = (
            action_history if support_history is None else support_history
        )
        if history_source is not None:
            history = history_source.expand(n, -1, -1)
            history_mask = torch.ones(
                history.shape[:2], dtype=torch.bool, device=history.device
            )
        support = self.model.core.action_support_head(
            expanded, support_codes, history, history_mask
        )
        prior = self.model.action_prior_head(expanded, action_codes)
        return (
            self.proposal_prior_weight * torch.log_softmax(prior, dim=0)
            + self.proposal_support_weight * torch.nn.functional.logsigmoid(
                support
            )
        )

    def _predicted_sequence_state(
        self,
        state: torch.Tensor,
        problem: Problem,
        sequence: list[int],
        state_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        codes = self._action_codes(problem, sequence).unsqueeze(0)
        if hasattr(self.model.predictor, "rollout"):
            return self.model.predictor.rollout(
                state,
                codes,
                state_history=state_history,
                action_history=action_history,
            )[:, -1]
        current = state
        for step in range(codes.shape[1]):
            current = self.model.predictor(current, codes[:, step])
        return current

    @torch.no_grad()
    def _learned_catalogue_sequences(
        self,
        state: torch.Tensor,
        problem: Problem,
        executed: list[int],
        state_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> tuple[list[list[int]], torch.Tensor]:
        """Build a root-balanced beam without symbolic feasibility queries.

        The fixed catalogue contains every outcome-free intent phrase stated
        by the prompt.  Only actions known to have executed are removed.  At
        imagined states, availability and usefulness are estimated entirely
        by learned heads applied to predicted latents.
        """
        executed_set = set(executed)
        ordered_catalogue = getattr(
            self,
            "_episode_catalogue",
            [variable.idx for variable in problem.vars],
        )
        if set(ordered_catalogue) != {variable.idx for variable in problem.vars}:
            raise RuntimeError("episode action catalogue does not match problem")
        catalogue = [
            action for action in ordered_catalogue if action not in executed_set
        ]
        if not catalogue:
            raise RuntimeError("learned action catalogue is empty")
        root_codes = self._action_codes(problem, catalogue)
        root_support_codes = self._support_codes(problem, catalogue)
        executed_support = self._support_codes(problem, executed).unsqueeze(0)
        root_scores = self._proposal_scores(
            state,
            root_codes,
            action_history,
            root_support_codes,
            executed_support,
        )
        root_n = min(self.proposal_top_m, len(catalogue))
        if self.max_expand < root_n:
            raise ValueError(
                "max_expand must retain at least one sequence per proposed root"
            )
        beam_per_root = min(
            self.proposal_beam_width,
            max(1, self.max_expand // root_n),
        )
        root_rows = root_scores.topk(root_n).indices.tolist()
        roots = [catalogue[row] for row in root_rows]
        root_costs = [-float(root_scores[row]) for row in root_rows]
        by_root: dict[int, list[tuple[list[int], float]]] = {
            root: [([root], cost)] for root, cost in zip(roots, root_costs)
        }
        for _depth in range(1, self.lookahead):
            next_by_root: dict[int, list[tuple[list[int], float]]] = {}
            for root, frontier in by_root.items():
                expanded: list[tuple[list[int], float]] = []
                for sequence, cumulative_cost in frontier:
                    predicted = self._predicted_sequence_state(
                        state,
                        problem,
                        sequence,
                        state_history,
                        action_history,
                    )
                    used = executed_set | set(sequence)
                    candidates = [
                        action for action in ordered_catalogue if action not in used
                    ]
                    if not candidates:
                        expanded.append((sequence, cumulative_cost))
                        continue
                    codes = self._action_codes(problem, candidates)
                    support_codes = self._support_codes(problem, candidates)
                    sequence_history = self._action_codes(
                        problem, sequence
                    ).unsqueeze(0)
                    proposal_history = torch.cat([
                        action_history, sequence_history
                    ], dim=1)
                    proposal_support_history = self._support_codes(
                        problem, list(executed) + sequence
                    ).unsqueeze(0)
                    scores = self._proposal_scores(
                        predicted,
                        codes,
                        proposal_history,
                        support_codes,
                        proposal_support_history,
                    )
                    keep_n = min(self.proposal_top_m, len(candidates))
                    for row in scores.topk(keep_n).indices.tolist():
                        expanded.append((
                            sequence + [candidates[row]],
                            cumulative_cost - float(scores[row]),
                        ))
                expanded.sort(key=lambda item: item[1])
                next_by_root[root] = expanded[:beam_per_root]
            by_root = next_by_root

        candidates = [item for frontier in by_root.values() for item in frontier]
        if not candidates:
            raise RuntimeError("learned action catalogue produced no sequences")
        sequences = [sequence for sequence, _ in candidates]
        costs = torch.tensor(
            [cost for _, cost in candidates],
            dtype=state.dtype,
            device=state.device,
        )
        return sequences, costs

    def _apply_action_prior(
        self, state: torch.Tensor, problem: Problem, seqs: list[list[int]]
    ) -> list[list[int]]:
        """Filter root proposals with a normalized candidate-scoring prior."""
        if not (self.prior_top_m or self.prior_only):
            return seqs
        roots = sorted({sequence[0] for sequence in seqs if sequence})
        if not roots:
            return seqs
        from textjepa.data.igsm.render import action_phrase

        tokens = self._tokens([action_phrase(problem, idx) for idx in roots])
        logits = self.model.score_action_prior(
            state, tokens.squeeze(0).unsqueeze(0)
        ).squeeze(0)
        keep_n = 1 if self.prior_only else min(self.prior_top_m, len(roots))
        selected = logits.topk(keep_n).indices.tolist()
        keep = {roots[i] for i in selected}
        necessary = set(problem.query_ancestors) & set(roots)
        self.prior_decisions += 1
        self.prior_necessary_recall_sum += (
            len(necessary & keep) / max(len(necessary), 1)
        )
        return [sequence for sequence in seqs if sequence and sequence[0] in keep]

    def _s0(self, prompt_tokens, prompt_mask) -> torch.Tensor:
        if not hasattr(self, "_s0_cache") or self._s0_cache[0] is not prompt_tokens:
            empty = torch.full(
                (1, 1, 1), self.vocab.pad_id, dtype=torch.long, device=self.device
            )
            no_steps = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
            s0, _ = self.model.encode_states(
                prompt_tokens, prompt_mask, empty, no_steps
            )
            self._s0_cache = (prompt_tokens, s0)
        return self._s0_cache[1]

    def _current_state(self, prompt_tokens, prompt_mask, step_texts) -> torch.Tensor:
        if not step_texts:
            return self._s0(prompt_tokens, prompt_mask)
        return self._encode_steps(prompt_tokens, prompt_mask, step_texts)

    @torch.no_grad()
    def _oracle_goal_state(
        self, problem: Problem, prompt_tokens, prompt_mask
    ) -> torch.Tensor:
        """Encode the solved terminal state (LeWM-style oracle diagnostic).

        Uses the ground-truth minimal solution, so this energy is an upper
        bound / distillation target, not a deployable planner.
        """
        env = SymbolicEnv(problem)
        texts = []
        while not env.solved:
            necessary = [
                a for a in env.feasible_actions() if a in problem.query_ancestors
            ]
            texts.append(env.step(min(necessary)))
        return self._encode_steps(prompt_tokens, prompt_mask, texts)

    def _encode_steps(self, prompt_tokens, prompt_mask, step_texts) -> torch.Tensor:
        step_tokens = self._tokens(step_texts)
        step_mask = torch.ones(
            1, len(step_texts), dtype=torch.bool, device=self.device
        )
        _, states = self.model.encode_states(
            prompt_tokens, prompt_mask, step_tokens, step_mask
        )
        return states[:, -1]

    def _causal_history(
        self,
        prompt_tokens: torch.Tensor,
        prompt_mask: torch.Tensor,
        step_texts: list[str],
        problem: Problem,
        action_history: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return teacher-forced ``s_0...s_t`` and ``a_0...a_{t-1}``."""
        s0 = self._s0(prompt_tokens, prompt_mask)
        if step_texts:
            step_tokens = self._tokens(step_texts)
            step_mask = torch.ones(
                1, len(step_texts), dtype=torch.bool, device=self.device
            )
            _, observed = self.model.encode_states(
                prompt_tokens, prompt_mask, step_tokens, step_mask
            )
            states = torch.cat([s0.unsqueeze(1), observed], dim=1)
        else:
            states = s0.unsqueeze(1)
        if action_history:
            actions = self._action_codes(problem, action_history).unsqueeze(0)
        else:
            actions = states.new_zeros(1, 0, self.model.core.d_action)
        return states, actions

    def _energy(
        self, cur: torch.Tensor, s0: torch.Tensor, steps: torch.Tensor,
        goal_state: torch.Tensor | None,
    ) -> torch.Tensor:
        n = cur.shape[0]
        if goal_state is not None:
            geo = getattr(self.model.core, "geo_head", None)
            fin, goal = (geo(cur), geo(goal_state)) if geo is not None else (cur, goal_state)
            ln = lambda x: torch.nn.functional.layer_norm(x, x.shape[-1:])
            return (ln(fin) - ln(goal)).abs().mean(-1)
        return steps + self.model.value_head(cur, s0.expand(n, -1))

    def _action_codes(self, problem: Problem, idxs: list[int]) -> torch.Tensor:
        from textjepa.data.igsm.render import action_phrase

        texts = [action_phrase(problem, i) for i in idxs]
        tokens = self._tokens(texts).squeeze(0).unsqueeze(1)  # [n, 1, L]
        return self.model.encode_actions(tokens).squeeze(1)

    def _support_codes(self, problem: Problem, idxs: list[int]) -> torch.Tensor:
        if getattr(self.model, "action_support_kind", "pairwise") != "history_attention":
            return self._action_codes(problem, idxs)
        from textjepa.data.igsm.render import action_phrase

        if not idxs:
            width = self.model.chunk_encoder.norm.normalized_shape[0]
            return torch.empty(0, width, device=self.device)
        texts = [action_phrase(problem, i) for i in idxs]
        tokens = self._tokens(texts).squeeze(0).unsqueeze(1)
        return self.model.encode_chunks(tokens).squeeze(1)

    def _flat_costs(
        self,
        s: torch.Tensor,
        s0: torch.Tensor,
        problem: Problem,
        seqs: list[list[int]],
        goal_state: torch.Tensor | None,
        state_history: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = len(seqs)
        total = torch.empty(n, device=self.device)
        for length in sorted({len(q) for q in seqs}):
            selected = [i for i, q in enumerate(seqs) if len(q) == length]
            if length == 0:
                cur = s.expand(len(selected), -1)
            else:
                flat = [a for i in selected for a in seqs[i]]
                future = self._action_codes(problem, flat).reshape(
                    len(selected), length, -1
                )
                if hasattr(self.model.predictor, "rollout"):
                    cur = self.model.predictor.rollout(
                        s.expand(len(selected), -1),
                        future,
                        state_history=(
                            state_history.expand(len(selected), -1, -1)
                            if state_history is not None else None
                        ),
                        action_history=(
                            action_history.expand(len(selected), -1, -1)
                            if action_history is not None else None
                        ),
                    )[:, -1]
                else:
                    cur = s.expand(len(selected), -1)
                    for step in range(length):
                        cur = self.model.predictor(cur, future[:, step])
            steps = torch.full(
                (len(selected),), float(length), device=self.device
            )
            total[torch.tensor(selected, device=self.device)] = self._energy(
                cur, s0, steps, goal_state
            )
        return total

    def _macro_costs(
        self,
        s: torch.Tensor,
        s0: torch.Tensor,
        problem: Problem,
        seqs: list[list[int]],
        goal_state: torch.Tensor | None,
    ) -> torch.Tensor:
        """Score (m*K)-step sequences with chained F_hi macro jumps (HWM)."""
        # Flat configurations intentionally set macro_k=0 so unused hierarchy
        # modules are absent/frozen. They must still pass through the ordinary
        # flat planner without a division-by-zero in this hierarchy gate.
        K = max(int(self.model.core.macro_k), 1)
        L = len(seqs[0])
        n = len(seqs)
        a = self._action_codes(
            problem, [i for q in seqs for i in q]
        ).reshape(n, L, -1)
        cur = s.expand(n, -1)
        for w in range(L // K):
            m = self.model.core.macro_encoder(a[:, w * K : (w + 1) * K])
            cur = self.model.core.hi_predictor(cur, m)
        steps = torch.full((n,), float(L), device=self.device)
        return self._energy(cur, s0, steps, goal_state)

    def _symbolic_costs(
        self,
        env,
        step_texts: list[str],
        s0: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_mask: torch.Tensor,
        seqs: list[list[int]],
        goal_state: torch.Tensor | None,
    ) -> torch.Tensor:
        """Upper-bound control: execute each candidate sequence in the
        SYMBOLIC environment, encode the true resulting state, apply the
        learned energy — no latent imagination at all."""
        all_texts = []
        for q in seqs:
            c = env.clone()
            all_texts.append(step_texts + [c.step(i) for i in q])
        n = len(all_texts)
        C = max(len(t) for t in all_texts)
        L = max(
            (len(self.vocab.encode(x)) for t in all_texts for x in t), default=1
        )
        tokens = torch.full(
            (n, C, L), self.vocab.pad_id, dtype=torch.long, device=self.device
        )
        mask = torch.zeros(n, C, dtype=torch.bool, device=self.device)
        for i, t in enumerate(all_texts):
            for c_i, x in enumerate(t):
                ids = self.vocab.encode(x)
                tokens[i, c_i, : len(ids)] = torch.tensor(ids, device=self.device)
                mask[i, c_i] = True
        _, states = self.model.encode_states(
            prompt_tokens.expand(n, -1, -1), prompt_mask.expand(n, -1),
            tokens, mask,
        )
        last = mask.sum(dim=1) - 1
        cur = states[torch.arange(n, device=self.device), last]
        steps = torch.tensor(
            [float(len(q)) for q in seqs], device=self.device
        )
        return self._energy(cur, s0, steps, goal_state)

    def _best_sequence(
        self,
        s: torch.Tensor,
        s0: torch.Tensor,
        problem: Problem,
        seqs: list[list[int]],
        goal_state: torch.Tensor | None = None,
        state_history: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
        sym_ctx: tuple | None = None,  # (env, step_texts, prompt_t, prompt_m)
    ) -> list[int]:
        if self.simulator == "symbolic" and sym_ctx is not None:
            env, step_texts, pt, pm = sym_ctx
            total = self._symbolic_costs(
                env, step_texts, s0, pt, pm, seqs, goal_state
            )
            return seqs[int(total.argmin().item())]
        K = max(int(self.model.core.macro_k), 1)
        full_len = (max(len(q) for q in seqs) // K) * K
        full = (
            [q for q in seqs if len(q) == full_len]
            if self.hierarchy and full_len >= K
            else []
        )
        if full:
            rest = [q for q in seqs if len(q) != full_len]
            costs = [self._macro_costs(s, s0, problem, full, goal_state)]
            cands = list(full)
            if rest:
                costs.append(self._flat_costs(
                    s, s0, problem, rest, goal_state,
                    state_history, action_history,
                ))
                cands += rest
            total = torch.cat(costs)
            return cands[int(total.argmin().item())]
        total = self._flat_costs(
            s, s0, problem, seqs, goal_state,
            state_history, action_history,
        )
        return seqs[int(total.argmin().item())]
