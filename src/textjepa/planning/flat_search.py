"""Closed-loop planning with the flat-backbone intent JEPA on faithful iGSM.

Protocol (no scored budget): run until solved or a runaway cap of
``cap_mult`` x necessary steps; report success and the distribution of
steps used.  Candidate interfaces:

* ``feasible_menu``   roots = the environment's feasible intents (menu);
* ``full_catalogue``  roots = every intent of the problem minus the
                      planner's own attempted set; invalid intents are
                      executed as no-ops (invalid outcome sentence);
* ``ldad_cycle``      catalogue candidates ranked/filtered by LDAD cycle
                      score (decode a candidate's own phrase from the
                      imagined displacement), then Energy;
* ``codebook_ground`` k-means codebook over TRAINING action vectors
                      grounded to the nearest catalogue phrase (menu-free);
* ``prior_propose``   sample K intent phrases from the backbone's own
                      next-token head (greedy + nucleus), keep parseable
                      unique ones, ground to the environment's action text,
                      plan by Energy over them (menu-free, oracle executor);
* ``autonomous``      prior_propose + the backbone writes the outcome
                      sentence itself (the environment is consulted only to
                      grade the final goal, as in the free-generation LM eval).

Lookahead > 1 is oracle-free: deeper slots are drawn from the same root pool
(catalogue minus used), never from the environment's future menus.  They are
selected by the model's OWN endpoint energy in a beam (``branch`` best
continuations per surviving beam, ``max_expand`` beams kept), not sampled at
random -- random tails made deeper search strictly noisier than depth 1.
Scoring is the horizon-blind endpoint Energy E(s, imagined endpoint, s_0);
the first action of the argmin rollout is executed and the plan is recomputed
(MPC) after every executed step.
"""

from __future__ import annotations

import math
import random

import torch


from textjepa.utils.compute_counter import ComputeCounter
from textjepa.planning.evaluate import aggregate_episodes
from textjepa.planning.ldad_decode import phrase_log_probs
from textjepa.planning.search import EpisodeResult

CANDIDATE_INTERFACES = (
    "feasible_menu", "full_catalogue", "ldad_cycle", "codebook_ground",
    "prior_propose", "autonomous",
)
MENU_FREE_INTERFACES = (
    "full_catalogue", "ldad_cycle", "codebook_ground", "prior_propose",
    "autonomous",
)


def _percentile(vals: list, q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = min(len(s) - 1, int(math.ceil(q * len(s))) - 1)
    return float(s[max(idx, 0)])


def _last_int(text: str):
    ints = [t.rstrip(".;,") for t in text.split()
            if t.rstrip(".;,").lstrip("-").isdigit()]
    return int(ints[-1]) if ints else None


class FlatPlanner:
    def __init__(self, model, vocab, device, lookahead: int = 1,
                 max_expand: int = 64, candidate_interface: str = "feasible_menu",
                 branch: int = 4, aggregate: str = "mean_prefix",
                 scorer: str = "energy",
                 endpoints: str = "imagined",
                 cap_mult: float = 4.0, mask_attempted: bool = True,
                 prior_samples: int = 16, prior_top_p: float = 0.95,
                 prior_temperature: float = 1.3, prior_greedy: int = 1,
                 phrase_token_cap: int = 24, outcome_token_cap: int = 96,
                 codebook_k: int = 64, codebook_seed: int = 0,
                 prior_top_k: int = 0, max_len: int = 4096,
                 counter: ComputeCounter | None = None):
        if candidate_interface not in CANDIDATE_INTERFACES:
            raise ValueError(f"unknown candidate interface {candidate_interface!r}")
        self.model = model
        self.vocab = vocab
        self.device = device
        self.lookahead = int(lookahead)
        self.max_expand = int(max_expand)
        self.branch = int(branch)
        if aggregate not in {"endpoint", "mean_prefix"}:
            raise ValueError(f"unknown aggregate {aggregate!r}")
        self.aggregate = aggregate
        # DIAGNOSTIC ONLY (candidate-privileged / oracle rows; never a paper
        # headline): `scorer=oracle_distance` ranks by latent distance to the
        # encoded TRUE solved state, `endpoints=true` executes each candidate
        # sequence in a copy of the environment and encodes the REAL state.
        if scorer not in {"energy", "oracle_distance"}:
            raise ValueError(f"unknown scorer {scorer!r}")
        if endpoints not in {"imagined", "true"}:
            raise ValueError(f"unknown endpoints {endpoints!r}")
        self.scorer = scorer
        self.endpoints = endpoints
        self.oracle_diagnostic = (scorer != "energy" or endpoints != "imagined")
        self.candidate_interface = candidate_interface
        self.cap_mult = float(cap_mult)
        self.mask_attempted = bool(mask_attempted)
        self.prior_samples = int(prior_samples)
        self.prior_top_p = float(prior_top_p)
        self.prior_temperature = float(prior_temperature)
        self.prior_greedy = int(prior_greedy)
        self.phrase_token_cap = int(phrase_token_cap)
        self.outcome_token_cap = int(outcome_token_cap)
        self.codebook_k = int(codebook_k)
        self.codebook_seed = int(codebook_seed)
        self.prior_top_k = int(prior_top_k)
        self.max_len = int(max_len)
        self.codebook = None
        # Test-time-compute accounting (passive); see utils/compute_counter.py.
        self.counter = counter if counter is not None else ComputeCounter()
        if candidate_interface == "ldad_cycle" and model.observed_action_decoder is None:
            raise RuntimeError("ldad_cycle needs observed_action_ldad=true")

    # ----------------------------------------------------------- encoding
    @torch.no_grad()
    def _state(self, history: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        toks = torch.tensor(history[-self.max_len:], dtype=torch.long,
                            device=self.device).unsqueeze(0)
        h = self.model.encode(toks)
        self.counter.backbone(1, toks.shape[1])
        return h[0, -1], h[0]

    @torch.no_grad()
    def _codes(self, history: list[int], phrases: list[list[int]]) -> torch.Tensor:
        # One block-attention pass over prefix + all candidate phrases.
        ctx = len(history[-self.max_len:])
        self.counter.backbone(1, ctx + sum(len(p) for p in phrases))
        return self.model.encode_candidates_in_context(
            history[-self.max_len:], phrases, self.device
        )

    # ----------------------------------------------------------- proposer
    @torch.no_grad()
    def _decode_sentence(self, history: list[int], cap: int, greedy: bool,
                         gen: torch.Generator | None) -> list[int]:
        phrase: list[int] = []
        for _ in range(cap):
            ctx = (history + phrase)[-self.max_len:]
            toks = torch.tensor(ctx, dtype=torch.long, device=self.device).unsqueeze(0)
            logits = self.model.encoder(toks)[0, -1].float()
            self.counter.backbone(1, toks.shape[1], generated=1)
            logits[self.vocab.pad_id] = float("-inf")
            if greedy:
                nxt = int(logits.argmax().item())
            else:
                probs = torch.softmax(logits / self.prior_temperature, -1)
                sp, si = probs.sort(descending=True)
                keep = (sp.cumsum(0) - sp) < self.prior_top_p
                keep[0] = True
                sp = sp * keep
                sp = sp / sp.sum()
                pick = torch.multinomial(sp, 1, generator=gen)
                nxt = int(si[pick].item())
            phrase.append(nxt)
            if self.vocab.id_to_token[nxt].endswith("."):
                break
        return phrase

    @torch.no_grad()
    def _sample_phrases(self, history: list[int], n: int, n_greedy: int,
                        cap: int, gen: torch.Generator | None) -> list[list[int]]:
        """Decode ``n`` intent phrases in one batch (rows < n_greedy greedy,
        the rest nucleus samples); a phrase ends at a token ending in '.'."""
        ctx = history[-self.max_len:]
        toks = torch.tensor(ctx, dtype=torch.long, device=self.device).unsqueeze(0).expand(n, -1)
        phrases = [[] for _ in range(n)]
        done = torch.zeros(n, dtype=torch.bool, device=self.device)
        greedy = torch.arange(n, device=self.device) < n_greedy
        for _ in range(cap):
            logits = self.model.encoder(toks)[:, -1].float()
            self.counter.backbone(
                toks.shape[0], toks.shape[1],
                generated=int((~done).sum().item()),
            )
            logits[:, self.vocab.pad_id] = float("-inf")
            g_pick = logits.argmax(-1)
            probs = torch.softmax(logits / self.prior_temperature, -1)
            sp, si = probs.sort(descending=True, dim=-1)
            keep = (sp.cumsum(-1) - sp) < self.prior_top_p
            keep[:, 0] = True
            sp = sp * keep
            sp = sp / sp.sum(-1, keepdim=True)
            s_pick = si.gather(1, torch.multinomial(sp, 1, generator=gen)).squeeze(1)
            nxt = torch.where(greedy, g_pick, s_pick)
            for i in range(n):
                if not done[i]:
                    phrases[i].append(int(nxt[i]))
            ends = torch.tensor(
                [self.vocab.id_to_token[int(t)].endswith(".") for t in nxt.tolist()],
                device=self.device,
            )
            done = done | ends
            if bool(done.all()):
                break
            toks = torch.cat([toks, nxt.unsqueeze(1)], 1)[:, -self.max_len:]
        return phrases

    @torch.no_grad()
    def _propose(self, history: list[int], env, rng_seed: int,
                 stats: dict) -> list:
        """Sample intent phrases; return grounded unique actions (catalogue
        objects).  Grounding = exact match on the environment's action
        text (the menu-free executor contract of plan_lm free generation)."""
        gen = torch.Generator(device=self.device).manual_seed(rng_seed)
        by_text = {env.action_text(q): q for q in env.fp.params}
        out, seen_text = [], set()
        n_prop = n_parse = 0
        for phrase in self._sample_phrases(
            history, self.prior_samples, self.prior_greedy, self.phrase_token_cap, gen
        ):
            text = self.vocab.decode(phrase).strip()
            n_prop += 1
            if text in seen_text:
                continue
            seen_text.add(text)
            q = by_text.get(text)
            if q is None:
                continue
            n_parse += 1
            out.append(q)
        stats["n_proposed"] += n_prop
        stats["n_parseable"] += n_parse
        stats["n_unique"] += len(out)
        return out

    # ------------------------------------------------------------ pools
    @torch.no_grad()
    def fit_action_prior(self, problems: list) -> torch.Tensor:
        """codebook_ground: k-means over TRAINING action vectors encoded in
        the context of their own problem prompt."""
        from textjepa.planning.codebook import fit_codebook

        vecs = []
        for fp in problems:
            env = fp.make_env()
            history = [t for s in fp.prompt_sentences for t in self.vocab.encode(s)]
            phrases = [self.vocab.encode(env.action_text(q)) for q in fp.action_order]
            vecs.append(self._codes(history, phrases))
        self.codebook = fit_codebook(torch.cat(vecs, 0), k=self.codebook_k,
                                     seed=self.codebook_seed)
        return self.codebook

    def _roots(self, env, history: list[int], attempted: set,
               rng: random.Random, stats: dict) -> list:
        iface = self.candidate_interface
        mask = attempted if self.mask_attempted else set()
        if iface == "feasible_menu":
            roots = list(env.feasible_actions())
        elif iface in {"prior_propose", "autonomous"}:
            roots = [q for q in self._propose(history, env, rng.randrange(1 << 30), stats)
                     if q not in mask]
        else:
            roots = [q for q in env.fp.action_order if q not in mask]
        rng.shuffle(roots)
        return roots

    @torch.no_grad()
    def _filter_roots(self, roots: list, env, state, codes_by_action: dict):
        """ldad_cycle / codebook_ground: rank catalogue roots by cycle score or
        ground codebook entries to nearest catalogue vectors."""
        iface = self.candidate_interface
        if iface == "codebook_ground":
            if self.codebook is None:
                raise RuntimeError("fit_action_prior must run before codebook_ground")
            cat = torch.stack([codes_by_action[q] for q in roots])
            rows = torch.cdist(self.codebook.to(cat.dtype), cat).argmin(1).tolist()
            keep = []
            for r in rows:
                if r not in keep:
                    keep.append(r)
            roots = [roots[r] for r in keep]
        if iface in {"ldad_cycle", "codebook_ground"}:
            cat = torch.stack([codes_by_action[q] for q in roots])
            s = state.unsqueeze(0).expand(cat.shape[0], -1)
            logits = self.model.observed_action_decoder(self.model.predict(s, cat) - s)
            scores = phrase_log_probs(
                logits.float(), [self.vocab.encode(env.action_text(q)) for q in roots]
            )
            order = torch.argsort(scores, descending=True).tolist()
            roots = [roots[i] for i in order]
            if self.prior_top_k > 0:
                roots = roots[: self.prior_top_k]
        return roots

    def _sequences(self, roots: list, pool: list, rng: random.Random) -> list[list]:
        """Legacy random-tail expansion (kept for the ablation flag)."""
        if self.lookahead == 1:
            return [[r] for r in roots]
        total = max(self.max_expand, len(roots))
        quotient, remainder = divmod(total, len(roots))
        seqs = []
        for i, root in enumerate(roots):
            for _ in range(quotient + int(i < remainder)):
                seq = [root]
                for _d in range(1, self.lookahead):
                    rest = [a for a in pool if a not in seq]
                    seq.append(rng.choice(rest) if rest else None)
                seqs.append(seq)
        rng.shuffle(seqs)
        return seqs


    # ------------------------------------------- diagnostic (oracle) helpers
    @torch.no_grad()
    def _encode_batch(self, histories: list[list[int]]) -> torch.Tensor:
        """Last hidden state for a batch of token histories (right padded;
        causal attention makes the padding inert for earlier positions)."""
        hs = [h[-self.max_len:] for h in histories]
        L = max(len(h) for h in hs)
        toks = torch.full((len(hs), L), self.vocab.pad_id, dtype=torch.long,
                          device=self.device)
        for i, h in enumerate(hs):
            toks[i, : len(h)] = torch.tensor(h, dtype=torch.long, device=self.device)
        outs = []
        for st in range(0, len(hs), 64):
            chunk = toks[st: st + 64]
            keep = max(len(h) for h in hs[st: st + 64])
            h = self.model.encode(chunk[:, :keep])
            self.counter.backbone(chunk.shape[0], keep)
            idx = torch.tensor([len(x) - 1 for x in hs[st: st + 64]], device=self.device)
            outs.append(h[torch.arange(h.shape[0], device=self.device), idx])
        return torch.cat(outs, 0)

    @torch.no_grad()
    def _goal_vector(self, env, history: list[int]):
        """ORACLE: encode the state reached by completing the problem from
        here along necessary feasible actions (the reference solved state)."""
        probe = env.clone()
        hist = list(history)
        guard = 0
        while not probe.solved and guard < 64:
            feas = probe.feasible_actions()
            if not feas:
                break
            nxt = [q for q in feas if q in env.fp.necessary] or list(feas)
            q = sorted(nxt)[0]
            hist = hist + self.vocab.encode(probe.action_text(q))
            hist = hist + self.vocab.encode(probe.step(q))
            guard += 1
        return self._encode_batch([hist])[0], bool(probe.solved)

    @torch.no_grad()
    def _true_endpoints(self, seqs: list[list], env, history: list[int]):
        """ORACLE: execute each candidate sequence in a copy of the env
        (illegal actions are no-ops, as for the real executor) and encode the
        resulting REAL state."""
        hists = []
        for seq in seqs:
            probe = env.clone()
            hist = list(history)
            for a in seq:
                if a is None:
                    continue
                hist = hist + self.vocab.encode(probe.action_text(a))
                hist = hist + self.vocab.encode(probe.step_or_invalid(a))
            hists.append(hist)
        return self._encode_batch(hists)

    @torch.no_grad()
    def _score(self, seqs: list[list], state, s0, code_of: dict,
               hist_s: list, hist_a: list, horizon: float, env=None,
               history=None, goal=None, endpoints: str | None = None,
               _force_oracle: bool = False):
        """Rank a batch of rollouts (lower = better).  Default = the endpoint
        energy E(s, imagine(s, seq), s_0) on IMAGINED endpoints.  The oracle
        diagnostic rows swap the endpoint source and/or the scorer."""
        n = len(seqs)
        depth = max(len(q) for q in seqs)
        D = state.shape[-1]
        any_code = next(iter(code_of.values()))
        act = torch.zeros(n, depth, D, device=self.device, dtype=any_code.dtype)
        act_mask = torch.zeros(n, depth, dtype=torch.bool, device=self.device)
        for i, q in enumerate(seqs):
            for d, entry in enumerate(q):
                if entry is not None:
                    act[i, d] = code_of[entry]
                    act_mask[i, d] = True
        root = state.unsqueeze(0).expand(n, -1)
        if self.model.predictor_kind == "causal":
            hs = (torch.stack([s0] + hist_s, 0).unsqueeze(0).expand(n, -1, -1)
                  if hist_s else s0.view(1, 1, -1).expand(n, -1, -1))
            ha = (torch.stack(hist_a, 0).unsqueeze(0).expand(n, -1, -1)
                  if hist_a else act[:, :0])
            endpoint, prefixes = self.model.imagine(root, act, act_mask, hs, ha,
                                                    return_prefixes=True)
        else:
            endpoint, prefixes = self.model.imagine(root, act, act_mask,
                                                    return_prefixes=True)
        self.counter.latent(n, depth)
        src = endpoints or self.endpoints
        if src == "true":
            endpoint = self._true_endpoints(seqs, env, history)
        if self.scorer == "oracle_distance" or _force_oracle:
            return (endpoint.float() - goal.float().unsqueeze(0)).norm(dim=-1)
        init = s0.unsqueeze(0).expand(n, -1)
        if self.aggregate == "endpoint" or src == "true" or depth == 1:
            self.counter.bump("energy_forwards", n)
            return self.model.energy(root, endpoint, init, float(horizon))
        # Trajectory-mean energy over the imagined prefixes s_1..s_depth.  The
        # endpoint-only score is blind to WASTED first steps: an illegal (no-op)
        # action reaches the same endpoint for free, so at depth > 1 the argmin
        # is free to start with an illegal action.  Averaging over prefixes
        # charges every step.  Identical to the endpoint score at depth 1.
        self.counter.bump("energy_forwards", n * depth)
        es = [self.model.energy(root, prefixes[:, h + 1], init, float(h + 1))
              for h in range(depth)]
        return torch.stack(es, 1).mean(1)

    @torch.no_grad()
    def _beam(self, roots: list, pool: list, state, s0, code_of: dict,
              hist_s: list, hist_a: list, rng: random.Random, env=None,
              history=None, goal=None):
        """Energy-guided beam over imagined rollouts.  Depth 1 is exactly the
        old behaviour (score every root); each deeper level extends every
        surviving beam by its ``branch`` lowest-energy pool continuations and
        keeps the ``max_expand`` best beams overall."""
        seqs = [[r] for r in roots]
        kw = dict(env=env, history=history, goal=goal)
        energy = self._score(seqs, state, s0, code_of, hist_s, hist_a, 1, **kw)
        for d in range(2, self.lookahead + 1):
            order = torch.argsort(energy).tolist()[: self.max_expand]
            beams = [seqs[i] for i in order]
            cand = []
            for seq in beams:
                rest = [a for a in pool if a not in seq]
                if not rest:
                    cand.append(seq + [None])
                    continue
                ext = [seq + [a] for a in rest]
                e = self._score(ext, state, s0, code_of, hist_s, hist_a, d, **kw)
                keep = torch.argsort(e).tolist()[: max(self.branch, 1)]
                cand += [ext[i] for i in keep]
            seqs = cand
            energy = self._score(seqs, state, s0, code_of, hist_s, hist_a, d, **kw)
        return seqs, energy

    # ------------------------------------------------------------ episode
    @torch.no_grad()
    def plan_episode(self, fp, seed: int = 0) -> dict:
        env = fp.make_env()
        self.counter.new_episode()
        rng = random.Random(seed)
        history = [t for s in fp.prompt_sentences for t in self.vocab.encode(s)]
        prompt_len = len(history)
        s0, _ = self._state(history)
        n_necessary = len(fp.necessary)
        cap = int(math.ceil(self.cap_mult * n_necessary))
        steps = n_invalid = n_distr = 0
        attempted: set = set()
        stats = {"n_proposed": 0, "n_parseable": 0, "n_unique": 0, "n_kept": 0,
                 "n_no_proposal": 0, "recall_hits": 0, "recall_steps": 0,
                 "imagined_slots": 0, "imagined_invalid": 0,
                 "beam_slots": 0, "beam_better_than_d1": 0}
        menu_free = self.candidate_interface in MENU_FREE_INTERFACES
        autonomous = self.candidate_interface == "autonomous"
        value_match = n_valid = 0
        answer_correct = None
        hist_s: list[torch.Tensor] = []
        hist_a: list[torch.Tensor] = []
        while not env.solved and steps < cap:
            state, _ = self._state(history)
            roots = self._roots(env, history, attempted, rng, stats)
            if self.candidate_interface in {"prior_propose", "autonomous"}:
                stats["recall_steps"] += 1
                feas = set(env.feasible_actions())
                stats["recall_hits"] += int(any(q in feas for q in roots))
            if not roots and self.mask_attempted and attempted - set(env.resolved):
                # Every candidate at this state has been masked as invalid.
                # An intent that was illegal earlier can be legal again later,
                # so clear the invalid mask and retry instead of dead-ending;
                # the runaway cap (not the mask) is the budget.
                attempted = set(env.resolved)
                roots = self._roots(env, history, attempted, rng, stats)
            if not roots:
                stats["n_no_proposal"] += 1
                break
            catalogue = [q for q in fp.action_order]
            pool_actions = catalogue if self.candidate_interface != "prior_propose" and not autonomous else roots
            needed = list(dict.fromkeys(list(roots) + (pool_actions if self.lookahead > 1 else [])))
            phrases = [self.vocab.encode(env.action_text(q)) for q in needed]
            codes = self._codes(history, phrases)
            code_of = {q: codes[i] for i, q in enumerate(needed)}
            if self.candidate_interface in {"ldad_cycle", "codebook_ground"}:
                roots = self._filter_roots(roots, env, state, code_of)
                if not roots:
                    break
            stats["n_kept"] += len(roots)
            pool = [q for q in pool_actions if q not in attempted] if self.mask_attempted else list(pool_actions)
            goal = None
            if self.scorer == "oracle_distance" or self.lookahead > 1:
                # goal vector: ORACLE for the oracle_distance scorer, and the
                # measurement metric for the depth-offers-better-options stat.
                goal, _g_ok = self._goal_vector(env, history)
            if self.lookahead == 1:
                seqs = [[r] for r in roots]
                energy = self._score(seqs, state, s0, code_of, hist_s, hist_a,
                                     self.lookahead, env=env, history=history,
                                     goal=goal)
            else:
                seqs, energy = self._beam(roots, pool, state, s0, code_of,
                                          hist_s, hist_a, rng, env=env,
                                          history=history, goal=goal)
                # MEASUREMENT (oracle): does deeper imagination even offer
                # options whose imagined endpoint is closer to the solved
                # state than the depth-1 pick's would be?
                d1 = [[r] for r in roots]
                e1 = self._score(d1, state, s0, code_of, hist_s, hist_a, 1,
                                 env=env, history=history, goal=goal)
                pick1 = d1[int(e1.argmin().item())]
                dist1 = self._score([pick1], state, s0, code_of, hist_s, hist_a,
                                    1, env=env, history=history, goal=goal,
                                    _force_oracle=True)[0]
                dall = self._score(seqs, state, s0, code_of, hist_s, hist_a,
                                   self.lookahead, env=env, history=history,
                                   goal=goal, _force_oracle=True)
                stats["beam_slots"] += int(dall.numel())
                stats["beam_better_than_d1"] += int((dall < dist1).sum().item())
            best = seqs[int(energy.argmin().item())]
            q = best[0]
            if len(best) > 1:
                # Diagnostic only (oracle used for MEASUREMENT): would the
                # deeper imagined slots of the chosen rollout be legal when
                # reached?  Invalid slots stay no-ops, like the executor.
                probe = env.clone()
                for d_i, a_i in enumerate(best):
                    if a_i is None:
                        break
                    legal = a_i in probe.feasible_actions()
                    if d_i >= 1:
                        stats["imagined_slots"] += 1
                        stats["imagined_invalid"] += int(not legal)
                    if legal:
                        probe.step(a_i)
            n_distr += int(q not in fp.necessary)
            phrase = self.vocab.encode(env.action_text(q))
            history = history + phrase
            hist_a.append(code_of[q])
            steps += 1
            if menu_free:
                invalid = q not in env.feasible_actions()
                n_invalid += int(invalid)
                true_outcome = env.step_or_invalid(q)
                if invalid:
                    attempted.add(q)
                else:
                    attempted = set(env.resolved)
            else:
                invalid = False
                true_outcome = env.step(q)
                attempted = set(env.resolved)
            if autonomous:
                gen_out = self._decode_sentence(history, self.outcome_token_cap, greedy=True, gen=None)
                history = history + gen_out
                if not invalid:
                    n_valid += 1
                    ok = _last_int(self.vocab.decode(gen_out)) == _last_int(true_outcome)
                    value_match += int(ok)
                    if env.solved:
                        answer_correct = bool(ok and _last_int(self.vocab.decode(gen_out)) == fp.answer)
            else:
                history = history + self.vocab.encode(true_outcome)
            st, _ = self._state(history)
            hist_s.append(st)
        return {
            "solved": bool(env.solved), "steps": steps, "necessary": n_necessary,
            "n_distractor": n_distr, "n_invalid": n_invalid,
            "solved_at": steps if env.solved else None,
            "answer_correct": answer_correct,
            "value_match": value_match, "valid_steps": n_valid,
            "stats": stats,
        }


def _reference_episode(fp, policy: str, menu_free: bool, cap_mult: float,
                       rng: random.Random, mask: bool = True) -> dict:
    env = fp.make_env()
    cap = int(math.ceil(cap_mult * len(fp.necessary)))
    steps = n_d = n_inv = 0
    attempted: set = set()
    while not env.solved and steps < cap:
        if menu_free:
            cands = [q for q in fp.action_order if not (mask and q in attempted)]
        else:
            cands = env.feasible_actions()
        if not cands:
            break
        q = rng.choice(cands) if policy == "random" else cands[0]
        n_d += int(q not in fp.necessary)
        if menu_free:
            invalid = q not in env.feasible_actions()
            n_inv += int(invalid)
            env.step_or_invalid(q)
            attempted = attempted | {q} if invalid else set(env.resolved)
        else:
            env.step(q)
        steps += 1
    return {"solved": bool(env.solved), "steps": steps, "necessary": len(fp.necessary),
            "n_distractor": n_d, "n_invalid": n_inv,
            "solved_at": steps if env.solved else None}


def summarize(episodes: list[dict]) -> dict:
    n = max(len(episodes), 1)
    solved = [e for e in episodes if e["solved"]]
    total_steps = sum(e["steps"] for e in episodes)
    out = {
        "success": sum(e["solved"] for e in episodes) / n,
        "mean_steps": total_steps / n,
        "steps_median": _percentile([e["steps"] for e in episodes], 0.5),
        "steps_p90": _percentile([e["steps"] for e in episodes], 0.9),
        "mean_necessary": sum(e["necessary"] for e in episodes) / n,
        "solved_steps_over_necessary_mean": (
            sum(e["steps"] / e["necessary"] for e in solved) / len(solved) if solved else float("nan")),
        "solved_steps_over_necessary_median": _percentile(
            [e["steps"] / e["necessary"] for e in solved], 0.5),
        "solved_exact_necessary_frac": (
            sum(e["steps"] == e["necessary"] for e in solved) / len(solved) if solved else float("nan")),
        "distractor_rate": sum(e["n_distractor"] for e in episodes) / max(total_steps, 1),
        "invalid_action_rate": sum(e["n_invalid"] for e in episodes) / max(total_steps, 1),
        "n_episodes": len(episodes),
    }
    if any(e.get("stats") for e in episodes):
        st = [e["stats"] for e in episodes if e.get("stats")]
        rs = sum(s["recall_steps"] for s in st)
        if rs:
            out["proposal_recall"] = sum(s["recall_hits"] for s in st) / rs
            out["proposal_parse_rate"] = sum(s["n_parseable"] for s in st) / max(sum(s["n_proposed"] for s in st), 1)
            out["proposal_unique_per_step"] = sum(s["n_unique"] for s in st) / rs
        out["no_proposal_episode_rate"] = sum(s["n_no_proposal"] > 0 for s in st) / n
        bs = sum(s.get("beam_slots", 0) for s in st)
        if bs:
            out["beam_closer_to_goal_than_d1_frac"] = (
                sum(s.get("beam_better_than_d1", 0) for s in st) / bs)
        slots = sum(s["imagined_slots"] for s in st)
        if slots:
            out["imagined_invalid_rate"] = sum(s["imagined_invalid"] for s in st) / slots
    if any(e.get("answer_correct") is not None for e in episodes):
        out["success_answer_rate"] = sum(bool(e.get("answer_correct")) for e in episodes) / n
        vs = sum(e.get("valid_steps", 0) for e in episodes)
        out["outcome_value_match_rate"] = sum(e.get("value_match", 0) for e in episodes) / max(vs, 1)
    return out


def evaluate_flat_planning(planner: FlatPlanner, dataset, n_episodes: int,
                           seed: int = 0, log_every: int = 10) -> dict:
    menu_free = planner.candidate_interface in MENU_FREE_INTERFACES
    rng = random.Random(seed)
    planned, rand_, first_ = [], [], []
    for i in range(n_episodes):
        fp, _ = dataset.problem(i)
        planned.append(planner.plan_episode(fp, seed=seed + i))
        rand_.append(_reference_episode(fp, "random", menu_free, planner.cap_mult, rng, planner.mask_attempted))
        first_.append(_reference_episode(fp, "first", menu_free, planner.cap_mult, rng, planner.mask_attempted))
        if log_every and (i + 1) % log_every == 0:
            sr = sum(e["solved"] for e in planned) / len(planned)
            print(f"[ep {i+1}/{n_episodes}] success={sr:.3f}", flush=True)
    return {
        "compute": planner.counter.report(),
        "latent_planner": summarize(planned),
        "random_policy": summarize(rand_),
        "first_feasible_policy": summarize(first_),
        "episodes": planned,
        # Per-episode records for the reference policies too, so the
        # steps-to-solve *distribution* can be plotted for every policy on one
        # axis, not just aggregate means for the baselines.
        "episodes_random": rand_,
        "episodes_first_feasible": first_,
    }
