"""Closed-loop latent planning on the FAITHFUL iGSM domain.

Same protocol as the stylized planner: enumerate every currently feasible
parameter, encode its intent phrase, score F(s, a) with the value head,
execute the argmin in the official environment, re-encode, and replan.
Lookahead greater than one uses the reference environment to enumerate future
feasible sequences and is therefore an explicitly opt-in diagnostic.
"""

from __future__ import annotations

import math
import random

import torch

from textjepa.data.faithful import FaithfulDataset, FaithfulEnv
from textjepa.data.vocab import Vocab
from textjepa.planning.evaluate import aggregate_episodes
from textjepa.planning.ldad_decode import (
    delta_logits,
    encode_phrases,
    phrase_log_probs,
    require_ldad_decoder,
)
from textjepa.planning.search import EpisodeResult


#: Candidate interfaces this planner actually implements.  Anything else
#: must RAISE: silently falling back to the feasible menu is the defect this
#: module was fixed for (see CAMPAIGN_LOG 2026-08-13).
CANDIDATE_INTERFACES = (
    "feasible_menu", "full_catalogue", "ldad_cycle", "codebook_ground",
)

#: Interfaces that never see the environment's feasible menu: execution goes
#: through ``step_or_invalid`` with invalid counting, and reference policies
#: run over the full catalogue under the same attempted-mask.
MENU_FREE_INTERFACES = ("full_catalogue", "ldad_cycle", "codebook_ground")

#: Interfaces whose candidates are ranked by the LDAD cycle score.
CYCLE_INTERFACES = ("ldad_cycle", "codebook_ground")


def faithful_catalogue(env: FaithfulEnv, attempted=frozenset()) -> list:
    """Every action of the problem, with no feasibility filtering.

    Mirrors ``range(len(problem.vars))`` in the stylized planner: the whole
    variable catalogue, including already-resolved variables, in the
    problem's stable shuffled order (``FaithfulProblem.action_order``),
    minus ``attempted`` -- the policy's OWN already-tried actions, which is
    self-knowledge rather than an oracle signal.
    """
    return [q for q in env.fp.action_order if q not in attempted]


class FaithfulPlanner:
    def __init__(self, model, vocab: Vocab, device: torch.device,
                 lookahead: int = 1, max_expand: int = 64,
                 allow_oracle_future_actions: bool = False,
                 candidate_interface: str = "feasible_menu",
                 invalid_action_mode: str = "noop",
                 mask_attempted: bool = True,
                 slack_frac: float = 0.0,
                 prior_top_k: int = 0,
                 codebook_k: int = 64,
                 codebook_seed: int = 0):
        if candidate_interface not in CANDIDATE_INTERFACES:
            raise ValueError(
                f"unknown candidate interface: {candidate_interface!r}; "
                "faithful iGSM planning supports "
                + ", ".join(sorted(CANDIDATE_INTERFACES))
            )
        if invalid_action_mode not in {"noop", "failure"}:
            raise ValueError(
                f"unknown invalid action mode: {invalid_action_mode!r}"
            )
        if (
            lookahead > 1
            and candidate_interface == "feasible_menu"
            and not allow_oracle_future_actions
        ):
            raise ValueError(
                "lookahead > 1 enumerates future actions with the reference "
                "environment; set allow_oracle_future_actions=true only for "
                "a labeled oracle-action diagnostic"
            )
        if candidate_interface in CYCLE_INTERFACES:
            # RuntimeError (via require_ldad_decoder) if the checkpoint was
            # not trained with model.observed_action_ldad=true: the knob is
            # valid, the checkpoint is the wrong one.
            require_ldad_decoder(
                model, f"candidate_interface={candidate_interface}"
            )
        if int(codebook_k) < 1:
            raise ValueError("codebook_k must be positive")
        self.model = model
        self.vocab = vocab
        self.device = device
        self.lookahead = lookahead
        self.max_expand = max_expand
        self.allow_oracle_future_actions = allow_oracle_future_actions
        self.candidate_interface = candidate_interface
        self.invalid_action_mode = invalid_action_mode
        # Self-knowledge, not an oracle: the planner remembers which actions
        # it already ATTEMPTED (executed or echoed back invalid) and stops
        # re-proposing them.  Without this, invalid=noop leaves the state
        # unchanged and a deterministic argmin re-selects the same infeasible
        # action forever -- the "no-op proposal loop" the stylized menu-free
        # interfaces already mask against (search.py / codebook.py
        # ``executed``).  Kept as a flag so the unmasked lock-in stays
        # measurable as an ablation.  No effect under feasible_menu, where the
        # environment already removes resolved actions from the menu.
        self.mask_attempted = bool(mask_attempted)
        # Proportional slack: the fixed-slack ruler is NOT length-invariant
        # (slack 4 is 59% extra budget on a 7-step problem but 26% on a
        # 15-step one, which made the long-OOD band score HIGHER than ID for
        # every policy).  budget = necessary + slack + ceil(slack_frac *
        # necessary) keeps the cushion a constant fraction of the solution.
        self.slack_frac = float(slack_frac)
        # Cycle interfaces: keep the prior_top_k best-scoring candidates
        # (0 = all), mirroring the stylized planner's knob of the same name.
        self.prior_top_k = int(prior_top_k)
        self.codebook_k = int(codebook_k)
        self.codebook_seed = int(codebook_seed)
        # k-means codebook over TRAINING action embeddings; fitted once via
        # fit_action_prior (codebook_ground only).
        self.codebook: torch.Tensor | None = None

    def _tokens(self, texts: list[str]) -> torch.Tensor:
        ids = [self.vocab.encode(t) for t in texts]
        L = max(len(i) for i in ids)
        out = torch.full((1, len(ids), L), self.vocab.pad_id, dtype=torch.long)
        for c, i in enumerate(ids):
            out[0, c, : len(i)] = torch.tensor(i)
        return out.to(self.device)

    def _state(self, pt, pm, step_texts):
        if not step_texts:
            empty = torch.full((1, 1, 1), self.vocab.pad_id, dtype=torch.long,
                               device=self.device)
            return self.model.encode_states(
                pt, pm, empty,
                torch.zeros(1, 1, dtype=torch.bool, device=self.device),
            )[0]
        st = self._tokens(step_texts)
        sm = torch.ones(1, st.shape[1], dtype=torch.bool, device=self.device)
        return self.model.encode_states(pt, pm, st, sm)[1][:, -1]

    def _sequences(
        self, env: FaithfulEnv, rng: random.Random,
        attempted: frozenset = frozenset(),
    ) -> list[list]:
        """Balanced fixed-depth rollouts with absorbing terminal padding."""
        if self.candidate_interface in MENU_FREE_INTERFACES:
            return self._catalogue_sequences(env, rng, attempted)
        roots = list(env.feasible_actions())
        rng.shuffle(roots)
        if not roots:
            return [[None] * self.lookahead]
        if self.lookahead == 1:
            return [[root] for root in roots]
        total = max(self.max_expand, len(roots))
        quotient, remainder = divmod(total, len(roots))
        sequences = []
        for root_index, root in enumerate(roots):
            for _ in range(quotient + int(root_index < remainder)):
                candidate = env.clone()
                candidate.resolved.append(root)  # feasibility-only transition
                sequence = [root]
                for _step in range(1, self.lookahead):
                    if candidate.solved:
                        sequence.append(None)
                        continue
                    feasible = list(candidate.feasible_actions())
                    if not feasible:
                        sequence.append(None)
                        continue
                    action = rng.choice(feasible)
                    candidate.resolved.append(action)
                    sequence.append(action)
                sequences.append(sequence)
        rng.shuffle(sequences)
        return sequences

    def _catalogue_sequences(
        self, env: FaithfulEnv, rng: random.Random,
        attempted: frozenset = frozenset(),
    ) -> list[list]:
        """Oracle-free rollouts over the problem's whole action catalogue.

        No feasibility information is consulted anywhere: roots are every
        action of the problem the planner has not already ATTEMPTED (its own
        history, when ``mask_attempted``), and deeper slots are sampled from
        the same pool, so lookahead > 1 needs no reference environment.
        """
        roots = faithful_catalogue(env, attempted if self.mask_attempted
                                   else frozenset())
        rng.shuffle(roots)
        if not roots:
            return [[None] * self.lookahead]
        if self.lookahead == 1:
            return [[root] for root in roots]
        total = max(self.max_expand, len(roots))
        quotient, remainder = divmod(total, len(roots))
        sequences = []
        for root_index, root in enumerate(roots):
            for _ in range(quotient + int(root_index < remainder)):
                sequence = [root]
                for _step in range(1, self.lookahead):
                    # Never repeat an action inside one imagined rollout
                    # either: under noop semantics a repeat is a guaranteed
                    # dead slot.
                    pool = [a for a in roots if a not in sequence]
                    sequence.append(rng.choice(pool) if pool else None)
                sequences.append(sequence)
        rng.shuffle(sequences)
        return sequences

    @torch.no_grad()
    def fit_action_prior(self, problems: list) -> torch.Tensor:
        """Fit the codebook on TRAINING-problem action phrases only.

        Same hook name and contract as the stylized proposers
        (``codebook.CodebookCycleProposer.fit_prior`` via
        ``LatentPlanner.fit_action_prior``): callers must pass TRAINING
        problems -- evaluation problems would leak their action catalogue
        into the proposal distribution.  Checkpoints trained with
        ``model.action_codebook_k`` carry their own codes; those take
        precedence, exactly as in the stylized path.
        """
        from textjepa.planning.codebook import fit_codebook

        stored = getattr(self.model, "action_codebook", None)
        ready = getattr(self.model, "action_codebook_ready", None)
        if stored is not None and ready is not None and bool(ready):
            self.codebook = stored.detach().to(self.device)
            self.codebook_k = self.codebook.shape[0]
            return self.codebook
        phrases = [
            FaithfulEnv(fp).action_text(q)
            for fp in problems for q in fp.action_order
        ]
        if not phrases:
            raise ValueError("no training action phrases to fit on")
        self.codebook = fit_codebook(
            encode_phrases(self.model, self.vocab, self.device, phrases),
            k=self.codebook_k, seed=self.codebook_seed,
        )
        return self.codebook

    @torch.no_grad()
    def _episode_catalogue(self, env: FaithfulEnv) -> tuple[list, torch.Tensor, list]:
        """Per-episode candidate pool for the cycle interfaces.

        Returns (actions, their action codes, their phrase token ids).  For
        ``ldad_cycle`` the pool is the problem's whole catalogue
        (``fp.action_order``); for ``codebook_ground`` each fitted code is
        grounded to the NEAREST current-problem catalogue embedding
        (environment-side grounding, mirroring the stylized
        ``CodebookGroundProposer``) and the pool is the deduplicated grounded
        subset.  No feasibility information is consulted anywhere.
        """
        actions = list(env.fp.action_order)
        phrases = [env.action_text(q) for q in actions]
        codes = encode_phrases(self.model, self.vocab, self.device, phrases)
        if self.candidate_interface == "codebook_ground":
            if self.codebook is None:
                raise RuntimeError(
                    "fit_action_prior must be called with training problems "
                    "before codebook_ground planning (the evaluation "
                    "problems' phrases must not be used to fit the codebook)"
                )
            grounded_rows = torch.cdist(self.codebook, codes).argmin(1).tolist()
            keep: list[int] = []
            for row in grounded_rows:
                if row not in keep:
                    keep.append(row)
            actions = [actions[row] for row in keep]
            phrases = [phrases[row] for row in keep]
            codes = codes[torch.tensor(keep, device=codes.device)]
        token_ids = [self.vocab.encode(text) for text in phrases]
        return actions, codes, token_ids

    @torch.no_grad()
    def _cycle_ranked(
        self, state: torch.Tensor, pool: list[int],
        actions: list, codes: torch.Tensor, token_ids: list,
    ) -> list[int]:
        """Pool positions ranked by the shared LDAD cycle score at ``state``.

        Score(a) = mean token log-prob of a's OWN phrase under the LDAD
        displacement decoder applied to ``predictor(s, a) - s`` -- the exact
        scoring rule of the stylized ``_cycle_candidates``
        (``ldad_decode.delta_logits`` + ``phrase_log_probs``).  Keeps the
        prior_top_k best (0 = all).  No oracle is consulted.
        """
        index = torch.tensor(pool, device=codes.device)
        scores = phrase_log_probs(
            delta_logits(
                self.model, state, codes[index],
                context=f"candidate_interface={self.candidate_interface}",
            ),
            [token_ids[i] for i in pool],
        )
        order = [
            pool[i] for i in
            torch.argsort(scores, descending=True, stable=True).tolist()
        ]
        keep = self.prior_top_k if self.prior_top_k > 0 else len(order)
        return order[:keep]

    @torch.no_grad()
    def _cycle_sequences(
        self, state: torch.Tensor, attempted: frozenset,
        actions: list, codes: torch.Tensor, token_ids: list,
    ) -> list[list]:
        """Cycle-ranked rollouts; deeper slots never consult the environment.

        Roots are the cycle-ranked (top prior_top_k) candidates at the
        current state.  For lookahead > 1 each root is extended GREEDILY: the
        remaining candidates are re-scored with the same cycle rule at the
        JEPA-imagined state (predictor applied to the sequence so far), the
        best is appended, and so on -- one sequence per root.  This is a
        simplification of the stylized beam expansion (which re-ranks
        cycle candidates at every beam node); it keeps the FaithfulPlanner's
        balanced fixed-depth ``_sequences`` structure while remaining fully
        oracle-free.  Actions already in the sequence are masked, exactly as
        in ``_catalogue_sequences``.
        """
        mask = attempted if self.mask_attempted else frozenset()
        pool = [i for i, a in enumerate(actions) if a not in mask]
        if not pool:
            return [[None] * self.lookahead]
        roots = self._cycle_ranked(state, pool, actions, codes, token_ids)
        if self.lookahead == 1:
            return [[actions[r]] for r in roots]
        sequences = []
        for root in roots:
            sequence_rows = [root]
            imagined = self.model.predictor(
                state.reshape(1, -1), codes[root].unsqueeze(0)
            )
            for _step in range(1, self.lookahead):
                rest = [i for i in pool if i not in sequence_rows]
                if not rest:
                    sequence_rows.append(None)
                    continue
                nxt = self._cycle_ranked(
                    imagined, rest, actions, codes, token_ids
                )[0]
                sequence_rows.append(nxt)
                imagined = self.model.predictor(
                    imagined, codes[nxt].unsqueeze(0)
                )
            sequences.append([
                actions[row] if row is not None else None
                for row in sequence_rows
            ])
        return sequences

    @torch.no_grad()
    def plan_episode(self, fp, slack: int = 0, seed: int = 0) -> EpisodeResult:
        env = FaithfulEnv(fp)
        pt = self._tokens(fp.prompt_sentences)
        pm = torch.ones(1, pt.shape[1], dtype=torch.bool, device=self.device)
        step_texts: list[str] = []
        budget = (len(fp.necessary) + slack
                  + math.ceil(self.slack_frac * len(fp.necessary)))
        n_distr = 0
        n_invalid = 0
        attempted: set = set()
        s0 = self._state(pt, pm, [])
        cycle = self.candidate_interface in CYCLE_INTERFACES
        if cycle:
            # Fixed per problem: the (possibly codebook-grounded) candidate
            # pool, its action codes and phrase tokens, computed once.
            cat_actions, cat_codes, cat_tokens = self._episode_catalogue(env)
        while not env.solved and len(step_texts) < budget:
            s = self._state(pt, pm, step_texts) if step_texts else s0
            if cycle:
                seqs = self._cycle_sequences(
                    s, frozenset(attempted),
                    cat_actions, cat_codes, cat_tokens,
                )
            else:
                seqs = self._sequences(
                    env,
                    random.Random(f"{seed}:{len(step_texts)}:candidates"),
                    frozenset(attempted),
                )
            if seqs and seqs[0][0] is None:
                # Masking exhausted the catalogue: the episode stalls (counted
                # unsolved) rather than re-proposing a known-dead action.
                break
            n = len(seqs)
            depth = max(len(q) for q in seqs)
            cur = s.expand(n, -1).clone()
            # Constant across candidates; unlike the historical accumulated
            # path length, it cannot disclose which rollout solved early.
            cost = torch.full(
                (n,), float(self.lookahead), device=self.device
            )
            for d in range(depth):
                texts, alive = [], []
                for q in seqs:
                    # ``_sequences`` pads with None once a rollout absorbs
                    # (solved / no feasible action); those slots are dead.
                    entry = q[d] if d < len(q) else None
                    if entry is not None:
                        texts.append(env.action_text(entry))
                        alive.append(True)
                    else:
                        texts.append(".")
                        alive.append(False)
                a = self.model.encode_actions(
                    self._tokens(texts).squeeze(0).unsqueeze(1)
                ).squeeze(1)
                alive_t = torch.tensor(alive, device=self.device)
                nxt = self.model.predictor(cur, a)
                cur = torch.where(alive_t.unsqueeze(1), nxt, cur)
            total = cost + self.model.value_head(cur, s0.expand(n, -1))
            best = seqs[int(total.argmin().item())]
            q = best[0]
            n_distr += int(q not in fp.necessary)
            if self.candidate_interface in MENU_FREE_INTERFACES:
                # invalid = noop: the executor returns the invalid-outcome
                # sentence, the symbolic state is unchanged, and the attempt
                # is counted in invalid_action_rate.
                invalid = q not in env.feasible_actions()
                n_invalid += int(invalid)
                step_texts.append(env.step_or_invalid(q))
                if invalid:
                    # Mask only WHILE the state is unchanged: an action that
                    # is infeasible now may become feasible after progress
                    # (its dependencies resolve).  A permanent mask made any
                    # necessary action tried too early unrecoverable and
                    # drove full_catalogue success to 0 for EVERY policy.
                    attempted.add(q)
                    if self.invalid_action_mode == "failure":
                        break
                else:
                    attempted = {a for a in env.resolved}
            else:
                step_texts.append(env.step(q))
                attempted = {a for a in env.resolved}
        return EpisodeResult(
            env.solved, len(step_texts), len(fp.necessary), n_distr,
            n_invalid,
            solved_at=len(step_texts) if env.solved else None,
        )


def evaluate_faithful_planning(
    planner: FaithfulPlanner, dataset: FaithfulDataset, n_episodes: int,
    slack: int = 0, seed: int = 0, slack_curve: bool = False,
) -> dict[str, dict]:
    """Faithful iGSM planning metrics in the shared JSON shape.

    With ``slack_curve`` the episodes are run once at the generous budget
    ``necessary + slack`` and then scored at every smaller slack; the policy
    never reads its budget, so the run at slack ``s`` is a prefix of the run at
    the largest slack. The scalar metrics therefore describe the generous run,
    and ``success_by_slack[str(slack)]`` equals ``success``.
    """
    interface = getattr(planner, "candidate_interface", "feasible_menu")
    # Reference lines must run under EXACTLY the planner's rules, including
    # the attempted-action mask; otherwise the comparison flatters us.
    mask = bool(getattr(planner, "mask_attempted", True))
    rng = random.Random(seed)
    planned, rand_, first_ = [], [], []
    for i in range(n_episodes):
        fp, _ = dataset.problem(i)
        planned.append(planner.plan_episode(fp, slack=slack, seed=seed + i))
        budget = (len(fp.necessary) + slack
                  + math.ceil(getattr(planner, "slack_frac", 0.0)
                              * len(fp.necessary)))

        # Reference policies see exactly the same candidate interface.
        env = FaithfulEnv(fp)
        steps = n_d = n_inv = 0
        attempted: set = set()
        while not env.solved and steps < budget:
            if interface in MENU_FREE_INTERFACES:
                candidates = faithful_catalogue(
                    env, frozenset(attempted) if mask else frozenset()
                )
                if not candidates:
                    break
                q = rng.choice(candidates)
                invalid = q not in env.feasible_actions()
                n_inv += int(invalid)
                env.step_or_invalid(q)
                attempted = attempted | {q} if invalid else set(env.resolved)
            else:
                q = rng.choice(env.feasible_actions())
                env.step(q)
            n_d += int(q not in fp.necessary)
            steps += 1
        rand_.append(EpisodeResult(
            env.solved, steps, len(fp.necessary), n_d, n_inv,
            solved_at=steps if env.solved else None,
        ))

        env = FaithfulEnv(fp)
        steps = n_d = n_inv = 0
        attempted = set()
        while not env.solved and steps < budget:
            candidates = (
                faithful_catalogue(
                    env, frozenset(attempted) if mask else frozenset()
                )
                if interface in MENU_FREE_INTERFACES
                else env.feasible_actions()
            )
            if not candidates:
                break
            q = candidates[0]
            n_d += int(q not in fp.necessary)
            if interface in MENU_FREE_INTERFACES:
                invalid = q not in env.feasible_actions()
                n_inv += int(invalid)
                env.step_or_invalid(q)
                attempted = (attempted | {q} if invalid
                             else set(env.resolved))
            else:
                env.step(q)
            steps += 1
        first_.append(EpisodeResult(
            env.solved, steps, len(fp.necessary), n_d, n_inv,
            solved_at=steps if env.solved else None,
        ))

    def agg(rs):
        return aggregate_episodes(rs, slack_curve=slack_curve, slack=slack)

    return {
        "latent_planner": agg(planned),
        "random_policy": agg(rand_),
        "first_feasible_policy": agg(first_),
    }
