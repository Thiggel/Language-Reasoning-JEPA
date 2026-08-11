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
from textjepa.data.igsm.render import (
    action_phrase, parse_action_phrase, prompt_sentences,
)
from textjepa.data.vocab import Vocab


def validate_learned_catalogue_checkpoint(cfg) -> None:
    """Reject checkpoints that lack learned-catalogue training support."""
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
    if float(cfg.objective.get("action_feasibility", {}).get("weight", 0.0)) <= 0:
        errors.append("objective.action_feasibility.weight must be positive")
    if float(cfg.objective.get("action_prior", {}).get("weight", 0.0)) <= 0:
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
    # Mean (over steps) 1-based rank of the best ground-truth-necessary
    # feasible action under the learned action prior. Diagnostic only: it
    # reads oracle labels for measurement, never for candidate generation.
    prior_rank: float | None = None
    # generator_cycle diagnostics. ``proposal_recall`` is the fraction of steps
    # at which at least one TRULY feasible action was among the parsed
    # generator samples; it reads the environment's feasible set for
    # measurement only, never for proposal or scoring. ``parse_rate`` is the
    # fraction of sampled phrases that parsed into the action space.
    proposal_recall: float | None = None
    parse_rate: float | None = None


def _feasible(problem: Problem, resolved: frozenset[int]) -> list[int]:
    return [
        v.idx
        for v in problem.vars
        if v.idx not in resolved and all(p in resolved for p in v.parents)
    ]


def _sequences(
    problem: Problem,
    resolved: frozenset[int],
    depth: int,
    cap: int,
    rng: random.Random | None = None,
) -> list[list[int | None]]:
    """Balanced fixed-depth oracle-action rollouts.

    Every currently feasible first action receives the same continuation
    budget (up to one rollout of rounding error). Future feasible actions are
    sampled with a seeded RNG, and candidates are shuffled before scoring.
    Once the query is solved, ``None`` denotes an absorbing no-op for every
    remaining depth. Thus candidate shape, count, and score offset cannot
    reveal how early a rollout reached the goal.
    """
    if depth < 1 or cap < 1:
        raise ValueError("depth and cap must be positive")
    rng = rng or random.Random(0)
    roots = _feasible(problem, resolved)
    rng.shuffle(roots)
    if not roots:
        return [[None] * depth]
    if depth == 1:
        return [[action] for action in roots]

    # Never discard a currently feasible action merely because the rollout
    # budget is narrow. Extra budget is divided as evenly as possible.
    total = max(cap, len(roots))
    quotient, remainder = divmod(total, len(roots))
    sequences: list[list[int | None]] = []
    for root_index, root in enumerate(roots):
        n_rollouts = quotient + int(root_index < remainder)
        for _ in range(n_rollouts):
            sequence: list[int | None] = [root]
            rollout_resolved = resolved | {root}
            for _step in range(1, depth):
                if problem.query in rollout_resolved:
                    sequence.append(None)
                    continue
                feasible = _feasible(problem, rollout_resolved)
                if not feasible:
                    sequence.append(None)
                    continue
                action = rng.choice(feasible)
                sequence.append(action)
                rollout_resolved = rollout_resolved | {action}
            sequences.append(sequence)
    rng.shuffle(sequences)
    return sequences


class LatentPlanner:
    def __init__(
        self,
        model,
        vocab: Vocab,
        device: torch.device,
        lookahead: int = 1,
        max_expand: int = 64,
        energy: str = "value",  # value | oracle_goal | symbolic_distance
        hierarchy: bool = False,  # score K-step sequences with F_hi jumps
        simulator: str = "latent",  # "latent" (F rollouts) | "symbolic"
        allow_oracle_future_actions: bool = False,
        score_control: str = "model",  # model | shuffle | zero
        search_algorithm: str = "shooting",  # shooting | beam
        transition_energy_composition: str = "terminal",
        hybrid_local_pruning: bool = False,
        candidate_interface: str = "feasible_menu",
        invalid_action_mode: str = "noop",
        prior_top_k: int = 0,
        prior_top_p: float = 1.0,
        prior_feasibility_gate: bool = False,
        generator_samples: int = 16,
        generator_top_p: float = 1.0,
        generator_temperature: float = 1.0,
    ):
        if candidate_interface not in {
            "feasible_menu", "full_catalogue", "learned_catalogue",
            "ldad_cycle", "generator_cycle",
        }:
            raise ValueError(
                f"unknown candidate interface: {candidate_interface}"
            )
        if candidate_interface == "learned_catalogue" and (
            hierarchy or simulator != "latent"
        ):
            raise ValueError(
                "learned_catalogue planning supports only the flat latent "
                "simulator"
            )
        if prior_top_k < 0:
            raise ValueError("prior_top_k must be non-negative (0 = off)")
        if not 0.0 < prior_top_p <= 1.0:
            raise ValueError("prior_top_p must lie in (0, 1]")
        if generator_samples < 1:
            raise ValueError("generator_samples must be positive")
        self.generator_samples = int(generator_samples)
        self.generator_top_p = float(generator_top_p)
        self.generator_temperature = float(generator_temperature)
        self.prior_top_k = prior_top_k
        self.prior_top_p = prior_top_p
        self.prior_feasibility_gate = prior_feasibility_gate
        if (
            lookahead > 1
            and candidate_interface == "feasible_menu"
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
        if energy == "symbolic_distance" and simulator != "symbolic":
            raise ValueError(
                "symbolic_distance is an oracle diagnostic and requires "
                "simulator=symbolic"
            )
        if energy not in {"value", "oracle_goal", "symbolic_distance"}:
            raise ValueError(f"unknown energy: {energy}")
        if score_control not in {"model", "shuffle", "zero"}:
            raise ValueError(f"unknown score control: {score_control}")
        self.score_control = score_control
        if search_algorithm not in {"shooting", "beam", "root_balanced_beam"}:
            raise ValueError(f"unknown search algorithm: {search_algorithm}")
        self.search_algorithm = search_algorithm
        if transition_energy_composition not in {
            "cumulative", "terminal", "root"
        }:
            raise ValueError(
                "unknown transition Energy composition: "
                f"{transition_energy_composition}"
            )
        self.transition_energy_composition = transition_energy_composition
        self.hybrid_local_pruning = bool(hybrid_local_pruning)
        self.candidate_interface = candidate_interface
        self.invalid_action_mode = invalid_action_mode
        if self.hybrid_local_pruning and getattr(
            self.model, "geo_rank_score_mode", "value"
        ) != "horizon":
            raise ValueError(
                "hybrid local pruning requires a horizon-Energy checkpoint"
            )

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
        env = SymbolicEnv(problem, self.invalid_action_mode)
        prompt = prompt_sentences(problem, random.Random(seed))
        prompt_tokens = self._tokens(prompt)
        prompt_mask = torch.ones(1, len(prompt), dtype=torch.bool, device=self.device)
        step_texts: list[str] = []
        action_history: list[int] = []
        budget = problem.n_necessary_steps + slack
        n_distractor = 0
        n_invalid = 0
        prior_ranks: list[float] = []
        proposal_hits: list[float] = []
        parse_rates: list[float] = []
        goal_state = (
            self._oracle_goal_state(problem, prompt_tokens, prompt_mask)
            if self.energy == "oracle_goal"
            else None
        )

        while not env.solved and not env.failed and len(step_texts) < budget:
            s = self._current_state(prompt_tokens, prompt_mask, step_texts)
            s0 = self._s0(prompt_tokens, prompt_mask)
            state_history, action_codes = self._causal_history(
                prompt_tokens,
                prompt_mask,
                step_texts,
                problem,
                action_history,
            )
            if self.candidate_interface == "learned_catalogue":
                prior_ranks.append(
                    self._prior_rank_diagnostic(
                        problem, s, frozenset(env.resolved_set)
                    )
                )
            root_candidates = None
            if self.candidate_interface == "generator_cycle":
                # Sample and parse the open-ended proposals once per step:
                # the diagnostics below and the search must see the same set.
                parsed, n_samples, unparseable = self._generator_proposals(
                    problem, s
                )
                feasible = set(env.feasible_actions())
                proposal_hits.append(
                    float(bool(feasible & set(parsed)))
                )
                parse_rates.append(
                    1.0 - unparseable / max(n_samples, 1)
                )
                root_candidates = self._cycle_candidates(
                    problem, s, executed=frozenset(env.resolved_set),
                    candidates=parsed,
                )
            if (
                self.candidate_interface
                in {"learned_catalogue", "generator_cycle"}
                or self.search_algorithm in {"beam", "root_balanced_beam"}
            ):
                best = self._beam_search(
                    s, s0, problem, frozenset(env.resolved_set), goal_state,
                    state_history, action_codes,
                    score_seed=f"{seed}:{len(step_texts)}:scores",
                    root_candidates=root_candidates,
                )
            else:
                seqs = _sequences(
                    problem,
                    frozenset(env.resolved_set),
                    self.lookahead,
                    self.max_expand,
                    random.Random(f"{seed}:{len(step_texts)}:candidates"),
                )
                best = self._best_sequence(
                    s, s0, problem, seqs, goal_state,
                    state_history=state_history,
                    action_history=action_codes,
                    sym_ctx=(env, step_texts, prompt_tokens, prompt_mask),
                    score_seed=f"{seed}:{len(step_texts)}:scores",
                )
            chosen = best[0]
            if chosen is None:
                # No candidate at all: under generator_cycle the head can fail
                # to produce a single parseable proposal. The episode stalls
                # (counted as unsolved) rather than falling back to a menu.
                break
            n_distractor += int(chosen not in problem.query_ancestors)
            invalid = chosen not in env.feasible_actions()
            n_invalid += int(invalid)
            step_texts.append(
                env.step_or_invalid(chosen)
                if self.candidate_interface
                in {
                    "full_catalogue", "learned_catalogue", "ldad_cycle",
                    "generator_cycle",
                }
                else env.step(chosen)
            )
            action_history.append(chosen)

        return EpisodeResult(
            env.solved, len(step_texts), problem.n_necessary_steps,
            n_distractor, n_invalid,
            prior_rank=(
                sum(prior_ranks) / len(prior_ranks) if prior_ranks else None
            ),
            proposal_recall=(
                sum(proposal_hits) / len(proposal_hits)
                if proposal_hits else None
            ),
            parse_rate=(
                sum(parse_rates) / len(parse_rates) if parse_rates else None
            ),
        )

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
        if self.energy == "symbolic_distance":
            raise RuntimeError(
                "symbolic_distance must be evaluated from exact environment states"
            )
        if (
            getattr(self.model, "geo_rank_score_mode", "value") == "distance"
            and goal_state is None
        ):
            raise RuntimeError(
                "geometry-only GAR requires energy=oracle_goal; its terminal "
                "state is a labeled diagnostic, not a deployable planner"
            )
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

    # ------------------------------------------------------------------ #
    # learned action prior: catalogue candidates with no feasibility oracle
    # ------------------------------------------------------------------ #
    def _catalogue_codes(self, problem: Problem) -> torch.Tensor:
        """[V, d_action] embeddings of every catalogue intent phrase."""
        cache = getattr(self, "_catalogue_cache", None)
        if cache is None or cache[0] is not problem:
            codes = self._action_codes(
                problem, list(range(len(problem.vars)))
            )
            self._catalogue_cache = (problem, codes)
        return self._catalogue_cache[1]

    def _prior_log_probs(
        self,
        problem: Problem,
        state: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """log p(a | state) for every catalogue action ([V])."""
        prior = getattr(self.model, "action_prior", None)
        if prior is None:
            raise RuntimeError(
                "candidate_interface=learned_catalogue requires a checkpoint "
                "trained with model.action_prior=true (no GaussianActionPrior "
                "head found on the model)"
            )
        codes = self._catalogue_codes(problem)
        expanded = state.reshape(1, -1).expand(codes.shape[0], -1)
        log_probs = prior.log_prob(expanded, codes)
        if self.prior_feasibility_gate:
            # Soft gate by the LEARNED feasibility head (trained with
            # objective.action_feasibility): adds log sigma(logit). No
            # oracle is consulted; both scores come from the checkpoint.
            # History-attention heads additionally see the executed (or
            # beam-imagined) action codes — still no oracle.
            hist = hist_mask = None
            if history is not None and history.shape[-2] > 0:
                hist = history.expand(codes.shape[0], -1, -1)
                hist_mask = torch.ones(
                    codes.shape[0], history.shape[-2],
                    dtype=torch.bool, device=codes.device,
                )
            logits = self.model.core.action_support_head(
                expanded, codes, history=hist, history_mask=hist_mask
            )
            log_probs = log_probs + torch.nn.functional.logsigmoid(logits)
        return log_probs

    def _prior_candidates(
        self,
        problem: Problem,
        state: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> list[int]:
        """Catalogue actions ranked and filtered by the learned prior.

        Top-p (nucleus over the prior's catalogue-normalized probabilities)
        then top-k, both optional. No feasibility oracle is consulted; the
        environment handles proposed-but-infeasible actions downstream.
        """
        log_probs = self._prior_log_probs(problem, state, history=history)
        order = torch.argsort(log_probs, descending=True, stable=True)
        keep = len(order)
        if self.prior_top_p < 1.0:
            sorted_probs = log_probs[order].softmax(-1)
            cumulative = sorted_probs.cumsum(-1)
            keep = min(
                keep, int((cumulative < self.prior_top_p).sum().item()) + 1
            )
        if self.prior_top_k > 0:
            keep = min(keep, self.prior_top_k)
        return order[:keep].tolist()

    def _cycle_candidates(
        self,
        problem: Problem,
        state: torch.Tensor,
        executed: frozenset[int] = frozenset(),
        candidates: list[int] | None = None,
    ) -> list[int]:
        """Actions filtered by LDAD cycle-consistency (no oracle).

        Score(a) = mean token log-prob of a's own intent phrase under the
        LDAD displacement decoder applied to the predictor's imagined
        displacement for a. Feasible actions produce action-identifiable
        displacements (probe: AUC .94 on the LDAD recipe); infeasible ones
        decode to noise. Keeps prior_top_k best (0 = all). ``executed``
        masks the planner's OWN executed/imagined actions (no oracle:
        already-computed variables stay "computable" per the decoder, so
        without the mask the planner loops on no-op re-proposals).

        ``candidates`` restricts scoring to a given action subset (used by
        ``generator_cycle``, whose proposals come from the generator head
        instead of the catalogue); ``None`` scores the whole catalogue.
        """
        decoder = getattr(self.model, "observed_action_decoder", None)
        if decoder is None:
            raise RuntimeError(
                f"candidate_interface={self.candidate_interface} requires a "
                "checkpoint trained with model.observed_action_ldad=true"
            )
        catalogue = self._catalogue_codes(problem)
        if candidates is None:
            candidates = list(range(catalogue.shape[0]))
        if not candidates:
            return []
        codes = catalogue[
            torch.tensor(candidates, device=catalogue.device)
        ]
        V = len(candidates)
        flat = state.reshape(1, -1)
        nxt = self.model.predictor(flat.expand(V, -1), codes)
        logits = decoder(nxt - flat)
        scores = []
        for position, action in enumerate(candidates):
            ids = torch.tensor(
                self.vocab.encode(action_phrase(problem, action)),
                device=logits.device,
            )
            L = min(ids.shape[0], logits.shape[1])
            scores.append(-torch.nn.functional.cross_entropy(
                logits[position, :L], ids[:L], reduction="mean"
            ))
        order = [
            candidates[position] for position in torch.argsort(
                torch.stack(scores), descending=True, stable=True
            ).tolist()
            if candidates[position] not in executed
        ]
        keep = self.prior_top_k if self.prior_top_k > 0 else V
        return order[:keep]

    def _generator_proposals(
        self, problem: Problem, state: torch.Tensor
    ) -> tuple[list[int], int, int]:
        """Sample intent phrases from the generator head and parse them.

        Fully open-ended: the generator writes free token sequences from the
        current state, with no candidate catalogue and no feasibility oracle.
        Each sample is grounded with :func:`parse_action_phrase`; samples that
        do not denote a well-formed action of this problem are discarded.
        Returns (parsed action indices, #samples, #unparseable samples), all
        counted over the DEDUPLICATED samples the head returned.
        """
        phrases = self.model.generate_action_phrases(
            state,
            k=self.generator_samples,
            top_p=self.generator_top_p,
            temperature=self.generator_temperature,
        )
        parsed: list[int] = []
        unparseable = 0
        for ids in phrases:
            action = parse_action_phrase(problem, self.vocab.decode(ids))
            if action is None:
                unparseable += 1
            elif action not in parsed:
                parsed.append(action)
        return parsed, len(phrases), unparseable

    def _generator_candidates(
        self,
        problem: Problem,
        state: torch.Tensor,
        executed: frozenset[int] = frozenset(),
    ) -> list[int]:
        """Generator proposals, ranked by the LDAD cycle-consistency score."""
        parsed, _, _ = self._generator_proposals(problem, state)
        return self._cycle_candidates(
            problem, state, executed=executed, candidates=parsed
        )

    def _imagined_state(
        self, problem: Problem, s: torch.Tensor, sequence: list[int]
    ) -> torch.Tensor:
        """Roll the predictor over a partial action sequence ([1, D])."""
        codes = self._action_codes(problem, sequence)
        if hasattr(self.model.predictor, "rollout"):
            return self.model.predictor.rollout(
                s, codes.unsqueeze(0)
            )[:, -1]
        cur = s
        for step in range(codes.shape[0]):
            cur = self.model.predictor(cur, codes[step : step + 1])
        return cur

    def _prior_rank_diagnostic(
        self, problem: Problem, state: torch.Tensor, resolved: frozenset[int]
    ) -> float:
        """1-based prior rank of the best ground-truth next action.

        Oracle labels (feasible necessary actions) are used only to *measure*
        how highly the prior ranks a correct continuation — never to build
        the candidate set.
        """
        log_probs = self._prior_log_probs(problem, state)
        order = torch.argsort(log_probs, descending=True, stable=True).tolist()
        targets = {
            action
            for action in _feasible(problem, resolved)
            if action in problem.query_ancestors
        }
        if not targets:
            return float(len(order))
        return float(
            min(order.index(action) for action in targets) + 1
        )

    def _flat_costs(
        self,
        s: torch.Tensor,
        s0: torch.Tensor,
        problem: Problem,
        seqs: list[list[int | None]],
        goal_state: torch.Tensor | None,
        state_history: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
        score_mode_override: str | None = None,
    ) -> torch.Tensor:
        active = [[action for action in sequence if action is not None]
                  for sequence in seqs]
        # ``None`` is an absorbing terminal no-op.  Keep the requested search
        # horizon separate from the number of actions executed before the
        # terminal state.  Otherwise a horizon-conditioned Energy can detect
        # symbolic early termination from the shortened horizon argument.
        requested_horizons = [len(sequence) for sequence in seqs]
        n = len(active)
        total = torch.empty(n, device=self.device)
        score_mode = score_mode_override or getattr(
            self.model, "geo_rank_score_mode", "value"
        )
        groups = sorted({
            (len(q), requested_horizons[i]) for i, q in enumerate(active)
        })
        for length, requested_horizon in groups:
            selected = [
                i for i, q in enumerate(active)
                if len(q) == length
                and requested_horizons[i] == requested_horizon
            ]
            if length == 0:
                cur = s.expand(len(selected), -1)
            else:
                flat = [a for i in selected for a in active[i]]
                future = self._action_codes(problem, flat).reshape(
                    len(selected), length, -1
                )
                if score_mode == "direct":
                    if length == 1:
                        direct_state = s.expand(len(selected), -1)
                    elif hasattr(self.model.predictor, "rollout"):
                        direct_state = self.model.predictor.rollout(
                            s.expand(len(selected), -1), future[:, :-1],
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
                        direct_state = s.expand(len(selected), -1)
                        for step in range(length - 1):
                            direct_state = self.model.predictor(
                                direct_state, future[:, step]
                            )
                    direct_cost = self.model.core.direct_action_rank_head(
                        direct_state, s0.expand(len(selected), -1), future[:, -1]
                    )
                    total[torch.tensor(selected, device=self.device)] = (
                        float(self.lookahead) + direct_cost
                    )
                    continue
                if hasattr(self.model.predictor, "rollout"):
                    rollout_states = self.model.predictor.rollout(
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
                    )
                    cur = rollout_states[:, -1]
                else:
                    cur = s.expand(len(selected), -1)
                    rollout = []
                    for step in range(length):
                        cur = self.model.predictor(cur, future[:, step])
                        rollout.append(cur)
                    rollout_states = torch.stack(rollout, dim=1)
            if score_mode == "td_q" and length > 0:
                # Q(s, a) semantics: score the state before the final action
                # of the beam together with that action.  Absorbing ``None``
                # padding never contributes actions, so the requested search
                # depth stays hidden from the head (terminal safety).
                pre_final = (
                    s.expand(len(selected), -1) if length == 1
                    else rollout_states[:, -2]
                )
                q = self.model.core.td_q_head(
                    pre_final, future[:, -1], s0.expand(len(selected), -1)
                )
                total[torch.tensor(selected, device=self.device)] = -q
                continue
            if score_mode == "td_jepa" and length > 0:
                # Faithful TD-JEPA score: Q = T(z_pre_final, u(a_final),
                # tau(z_0))^T z_r, with the same pre-final-state convention
                # as td_q (no rollout needed at depth 1) and z_r the ridge-
                # regressed task-reward projection fitted after training.
                core = self.model.core
                if not bool(core.td_jepa_z_r_fitted):
                    raise RuntimeError(
                        "td_jepa planning requires a fitted task-reward "
                        "projection; call fit_td_jepa_reward_projection on "
                        "training batches first (scripts/plan.py does this)"
                    )
                pre_final = (
                    s.expand(len(selected), -1) if length == 1
                    else rollout_states[:, -2]
                )
                task = core.td_jepa_task_head(s0.expand(len(selected), -1))
                features = core.successor_feature_head(
                    pre_final, future[:, -1], task
                )
                q = features @ core.td_jepa_z_r
                total[torch.tensor(selected, device=self.device)] = -q
                continue
            if score_mode == "goal_head" and length > 0:
                # Takai-style predicted-goal energy: LN-L1 distance of the
                # imagined endpoint to g(z_0) — mirrors energy=oracle_goal
                # with the learned goal prediction instead of the oracle
                # terminal encoding.
                goal_hat = self.model.core.goal_head(
                    s0.expand(len(selected), -1)
                )
                ln = lambda x: torch.nn.functional.layer_norm(
                    x, x.shape[-1:]
                )
                total[torch.tensor(selected, device=self.device)] = (
                    (ln(cur) - ln(goal_hat)).abs().mean(-1)
                )
                continue
            if score_mode == "expectile_value" and length > 0:
                v = self.model.core.expectile_value_head(
                    cur, s0.expand(len(selected), -1)
                )
                total[torch.tensor(selected, device=self.device)] = -v
                continue
            if (
                score_mode == "horizon" and length > 0
            ):
                sequence_energy = self.model.core.horizon_energy_head(
                    s.expand(len(selected), -1),
                    cur,
                    s0.expand(len(selected), -1),
                    requested_horizon,
                )
                total[torch.tensor(selected, device=self.device)] = (
                    sequence_energy
                )
                continue
            if (
                score_mode == "transition" and length > 0
            ):
                previous = torch.cat([
                    s.expand(len(selected), -1).unsqueeze(1),
                    rollout_states[:, :-1],
                ], dim=1)
                step_energy = self.model.core.transition_energy_head(
                    previous, rollout_states, s0.expand(len(selected), -1)
                )
                if self.transition_energy_composition == "cumulative":
                    sequence_energy = step_energy.sum(dim=1)
                elif self.transition_energy_composition == "terminal":
                    sequence_energy = step_energy[:, -1]
                else:
                    sequence_energy = step_energy[:, 0]
                total[torch.tensor(selected, device=self.device)] = (
                    sequence_energy
                )
                continue
            steps = torch.full(
                (len(selected),), float(self.lookahead), device=self.device
            )
            total[torch.tensor(selected, device=self.device)] = self._energy(
                cur, s0, steps, goal_state
            )
        return total

    def _beam_search(
        self,
        s: torch.Tensor,
        s0: torch.Tensor,
        problem: Problem,
        resolved: frozenset[int],
        goal_state: torch.Tensor | None,
        state_history: torch.Tensor,
        action_history: torch.Tensor,
        score_seed: str,
        root_candidates: list[int] | None = None,
    ) -> list[int | None]:
        """True global beam search over JEPA-imagined continuations.

        ``root_candidates`` supplies the depth-1 candidate set when the caller
        already produced it (generator_cycle, whose sampling is stochastic and
        also feeds the per-step proposal diagnostics).
        """
        if root_candidates is not None:
            roots = root_candidates
        elif self.candidate_interface == "full_catalogue":
            roots = list(range(len(problem.vars)))
        elif self.candidate_interface == "learned_catalogue":
            roots = self._prior_candidates(problem, s, history=action_history)
        elif self.candidate_interface == "ldad_cycle":
            roots = self._cycle_candidates(problem, s, executed=resolved)
        else:
            roots = _feasible(problem, resolved)
        beam = [[action] for action in roots]
        if not beam:
            return [None]
        for depth in range(1, self.lookahead + 1):
            if depth > 1:
                expanded: list[list[int | None]] = []
                for sequence in beam:
                    if self.candidate_interface == "full_catalogue":
                        expanded.extend(
                            sequence + [action]
                            for action in range(len(problem.vars))
                        )
                        continue
                    if self.candidate_interface == "learned_catalogue":
                        # Prior-filter expansions from the JEPA-imagined state
                        # after the partial sequence; no oracle menu is used.
                        expanded.extend(
                            sequence + [action]
                            for action in self._prior_candidates(
                                problem,
                                self._imagined_state(problem, s, sequence),
                                history=torch.cat(
                                    [
                                        action_history,
                                        self._action_codes(
                                            problem, sequence
                                        ).unsqueeze(0),
                                    ],
                                    dim=1,
                                ),
                            )
                        )
                        continue
                    if self.candidate_interface in {
                        "ldad_cycle", "generator_cycle"
                    }:
                        if sequence[-1] is None:
                            expanded.append(sequence)
                            continue
                        expand_fn = (
                            self._cycle_candidates
                            if self.candidate_interface == "ldad_cycle"
                            else self._generator_candidates
                        )
                        cands = expand_fn(
                            problem,
                            self._imagined_state(problem, s, sequence),
                            executed=resolved | {
                                a for a in sequence if a is not None
                            },
                        )
                        if cands:
                            expanded.extend(
                                sequence + [action] for action in cands
                            )
                        else:
                            expanded.append(sequence + [None])
                        continue
                    reached = resolved | {
                        a for a in sequence if a is not None
                    }
                    if problem.query in reached:
                        expanded.append(sequence + [None])
                        continue
                    actions = _feasible(problem, frozenset(reached))
                    expanded.extend(sequence + [a] for a in actions)
                beam = expanded
            score_override = (
                "value"
                if self.hybrid_local_pruning and depth < self.lookahead
                else None
            )
            score_args = (
                s, s0, problem, beam, goal_state,
                state_history, action_history,
            )
            costs = (
                self._flat_costs(
                    *score_args, score_mode_override=score_override
                )
                if score_override is not None
                else self._flat_costs(*score_args)
            )
            if self.search_algorithm == "root_balanced_beam":
                indices = []
                roots = []
                for sequence in beam:
                    if sequence[0] not in roots:
                        roots.append(sequence[0])
                for root in roots:
                    group = [
                        index for index, sequence in enumerate(beam)
                        if sequence[0] == root
                    ]
                    order = torch.argsort(
                        costs[torch.tensor(group, device=self.device)],
                        stable=True,
                    )[: self.max_expand].tolist()
                    indices.extend(group[index] for index in order)
            else:
                keep = min(self.max_expand, len(beam))
                indices = torch.argsort(costs, stable=True)[:keep].tolist()
            beam = [beam[i] for i in indices]
        score_args = (
            s, s0, problem, beam, goal_state,
            state_history, action_history,
        )
        costs = (
            self._flat_costs(*score_args, score_mode_override="value")
            if self.hybrid_local_pruning and self.lookahead == 1
            else self._flat_costs(*score_args)
        )
        return beam[self._controlled_argmin(costs, score_seed)]

    def _macro_costs(
        self,
        s: torch.Tensor,
        s0: torch.Tensor,
        problem: Problem,
        seqs: list[list[int | None]],
        goal_state: torch.Tensor | None,
    ) -> torch.Tensor:
        """Score (m*K)-step sequences with chained F_hi macro jumps (HWM)."""
        # Flat configurations intentionally set macro_k=0 so unused hierarchy
        # modules are absent/frozen. They must still pass through the ordinary
        # flat planner without a division-by-zero in this hierarchy gate.
        K = max(int(self.model.core.macro_k), 1)
        if any(any(action is None for action in sequence) for sequence in seqs):
            raise ValueError("absorbing-padded sequences require flat scoring")
        L = len(seqs[0])
        n = len(seqs)
        a = self._action_codes(
            problem, [i for q in seqs for i in q if i is not None]
        ).reshape(n, L, -1)
        cur = s.expand(n, -1)
        for w in range(L // K):
            m = self.model.core.macro_encoder(a[:, w * K : (w + 1) * K])
            cur = self.model.core.hi_predictor(cur, m)
        steps = torch.full((n,), float(L), device=self.device)
        return self._energy(cur, s0, steps, goal_state)

    def _symbolic_costs(
        self,
        problem: Problem,
        env,
        step_texts: list[str],
        s: torch.Tensor,
        s0: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_mask: torch.Tensor,
        seqs: list[list[int | None]],
        goal_state: torch.Tensor | None,
    ) -> torch.Tensor:
        """Upper-bound control: execute each candidate sequence in the
        SYMBOLIC environment, encode the true resulting state, apply the
        learned energy — no latent imagination at all."""
        active = [[action for action in sequence if action is not None]
                  for sequence in seqs]
        if self.energy == "symbolic_distance":
            # Perfect cost-to-go control. It uses the hidden symbolic state and
            # is therefore never a deployable model score.
            costs = []
            for actions in active:
                clone = env.clone()
                for action in actions:
                    clone.step(action)
                costs.append(float(clone.remaining_necessary()))
            return torch.tensor(costs, device=self.device)
        if getattr(self.model, "geo_rank_score_mode", "value") == "direct":
            # The direct scorer consumes the exact predecessor state and the
            # final action. Using value_head here would silently evaluate an
            # untrained module for direct-ranker checkpoints.
            predecessor_texts = []
            final_actions = []
            for actions in active:
                clone = env.clone()
                predecessor_texts.append(
                    step_texts + [clone.step(action) for action in actions[:-1]]
                )
                final_actions.append(actions[-1])
            n = len(predecessor_texts)
            nonempty = [index for index, text in enumerate(predecessor_texts) if text]
            predecessor = s.expand(n, -1).clone()
            if nonempty:
                selected_texts = [predecessor_texts[index] for index in nonempty]
                C = max(len(text) for text in selected_texts)
                L = max(
                    len(self.vocab.encode(sentence))
                    for text in selected_texts for sentence in text
                )
                tokens = torch.full(
                    (len(nonempty), C, L), self.vocab.pad_id,
                    dtype=torch.long, device=self.device,
                )
                mask = torch.zeros(
                    len(nonempty), C, dtype=torch.bool, device=self.device
                )
                for row, text in enumerate(selected_texts):
                    for column, sentence in enumerate(text):
                        ids = self.vocab.encode(sentence)
                        tokens[row, column, :len(ids)] = torch.tensor(
                            ids, device=self.device
                        )
                        mask[row, column] = True
                _, states = self.model.encode_states(
                    prompt_tokens.expand(len(nonempty), -1, -1),
                    prompt_mask.expand(len(nonempty), -1), tokens, mask,
                )
                rows = torch.tensor(nonempty, device=self.device)
                predecessor[rows] = states[
                    torch.arange(len(nonempty), device=self.device),
                    mask.sum(1) - 1,
                ]
            action_codes = self._action_codes(problem, final_actions)
            score = self.model.core.direct_action_rank_head(
                predecessor, s0.expand(n, -1), action_codes
            )
            return float(self.lookahead) + score

        all_texts = []
        for q in active:
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
        if getattr(self.model, "geo_rank_score_mode", "value") == "transition":
            if any(not actions for actions in active):
                raise ValueError(
                    "exact transition Energy requires a non-empty sequence"
                )
            predecessor = s.expand(n, -1).clone()
            multi_step = torch.tensor(
                [len(actions) > 1 for actions in active],
                dtype=torch.bool,
                device=self.device,
            )
            if multi_step.any():
                rows = torch.arange(n, device=self.device)[multi_step]
                predecessor[multi_step] = states[
                    rows, last[multi_step] - 1
                ]
            return self.model.core.transition_energy_head(
                predecessor, cur, s0.expand(n, -1)
            )
        steps = torch.tensor(
            [float(self.lookahead) for _ in seqs], device=self.device
        )
        return self._energy(cur, s0, steps, goal_state)

    def _best_sequence(
        self,
        s: torch.Tensor,
        s0: torch.Tensor,
        problem: Problem,
        seqs: list[list[int | None]],
        goal_state: torch.Tensor | None = None,
        state_history: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
        sym_ctx: tuple | None = None,  # (env, step_texts, prompt_t, prompt_m)
        score_seed: str = "0",
    ) -> list[int | None]:
        if self.simulator == "symbolic" and sym_ctx is not None:
            env, step_texts, pt, pm = sym_ctx
            total = self._symbolic_costs(
                problem, env, step_texts, s, s0, pt, pm, seqs, goal_state
            )
            return seqs[self._controlled_argmin(total, score_seed)]
        K = max(int(self.model.core.macro_k), 1)
        active_lengths = [sum(action is not None for action in q) for q in seqs]
        full_len = (max(active_lengths) // K) * K
        full = (
            [q for q, length in zip(seqs, active_lengths)
             if length == full_len and all(action is not None for action in q)]
            if self.hierarchy and full_len >= K
            else []
        )
        if full:
            rest = [q for q in seqs if q not in full]
            costs = [self._macro_costs(s, s0, problem, full, goal_state)]
            cands = list(full)
            if rest:
                costs.append(self._flat_costs(
                    s, s0, problem, rest, goal_state,
                    state_history, action_history,
                ))
                cands += rest
            total = torch.cat(costs)
            return cands[self._controlled_argmin(total, score_seed)]
        total = self._flat_costs(
            s, s0, problem, seqs, goal_state,
            state_history, action_history,
        )
        return seqs[self._controlled_argmin(total, score_seed)]

    def _controlled_argmin(self, costs: torch.Tensor, seed: str) -> int:
        """Apply an enumeration-leakage control before candidate selection."""
        if self.score_control == "zero":
            return 0
        if self.score_control == "shuffle":
            permutation = list(range(len(costs)))
            random.Random(seed).shuffle(permutation)
            index = torch.tensor(permutation, device=costs.device)
            return int(costs[index].argmin().item())
        return int(costs.argmin().item())
