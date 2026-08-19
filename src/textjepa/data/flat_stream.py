"""Flat token streams for the flat-backbone intent JEPA.

One causal token transformer reads the whole trace exactly as the token LM
does (``prompt, intent_1, outcome_1, ..., intent_T, outcome_T``).  This
module turns a :class:`textjepa.data.faithful.FaithfulDataset` item (token
lists per sentence, plus the geometric-ranking counterfactuals and rollouts)
into the tensors the flat model consumes:

* the main stream with boundary positions (``s_pos`` = hidden after each
  outcome, ``a_pos`` = hidden at the last intent token);
* the problem's action catalogue appended as a *phrase block* whose tokens
  attend only to the stream prefix up to the ranking anchor and causally
  within their own phrase -- this is how candidate intent phrases are
  encoded IN CONTEXT with one forward pass (same trick the planner uses);
* EMA-teacher token streams for the true counterfactual continuations and
  the random rollouts (labels of the endpoint Energy), rebuilt in the
  interleaved intent/outcome format.

Nothing symbolic is exported: every tensor is token text or an index into
token text.  ``ga_alt_invalid`` only records whether a counterfactual's
executor outcome was the literal "invalid definition" sentence (observable
from the outcome text itself).
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from textjepa.data.faithful import INVALID_DEFINITION_OUTCOME, FaithfulDataset


def _cat(lists: list[list[int]]) -> list[int]:
    return [tok for chunk in lists for tok in chunk]


class FlatIntentStreamDataset(Dataset):
    """Wrap a FaithfulDataset built with ``all_action_supervision=True``."""

    def __init__(self, base, lm_loss_on: str = "all_solution"):
        if lm_loss_on not in {"intent", "all_solution"}:
            raise ValueError(f"unknown lm_loss_on: {lm_loss_on}")
        if not getattr(base, "all_action_supervision", False):
            raise ValueError(
                "FlatIntentStreamDataset needs the action catalogue "
                "(dataset built with all_action_supervision=True)"
            )
        self.base = base
        self.lm_loss_on = lm_loss_on
        self.vocab = base.vocab
        # faithful and stylized iGSM render different "invalid action"
        # sentences; both datasets declare theirs as INVALID_OUTCOME.
        invalid = getattr(base, "INVALID_OUTCOME", INVALID_DEFINITION_OUTCOME)
        self._invalid_tokens = tuple(base.vocab.encode(invalid))

    def __len__(self) -> int:
        return len(self.base)

    def problem(self, index: int):
        return self.base.problem(index)

    def __getitem__(self, index: int) -> dict:
        item = self.base[index]
        prompt = _cat(item["prompt"])
        tokens = list(prompt)
        lm_mask = [False] * len(prompt)
        s_pos = [len(prompt) - 1]
        a_pos = []
        for action, outcome in zip(item["actions"], item["steps"]):
            tokens.extend(action)
            lm_mask.extend([True] * len(action))
            a_pos.append(len(tokens) - 1)
            tokens.extend(outcome)
            lm_mask.extend([self.lm_loss_on == "all_solution"] * len(outcome))
            s_pos.append(len(tokens) - 1)
        catalogue = [list(p) for p in item["action_candidate_tokens"]]
        cat_index = {tuple(p): i for i, p in enumerate(catalogue)}
        out = {
            "tokens": tokens,
            "lm_mask": lm_mask,
            "s_pos": s_pos,
            "a_pos": a_pos,
            "action_tokens": [list(a) for a in item["actions"]],
            "catalogue": catalogue,
            "index": index,
            "n_necessary": item["n_necessary"],
        }
        if "ga_t" in item:
            t = int(item["ga_t"])
            anchor_pos = s_pos[t]
            prefix = tokens[: anchor_pos + 1]
            cand_cat = [cat_index[tuple(item["actions"][t])]]
            cand_cat += [cat_index[tuple(a)] for a in item["ga_alt_actions"]]
            alt_invalid = [
                tuple(st) == self._invalid_tokens for st in item["ga_alt_steps"]
            ]
            roll_tokens, roll_first, roll_leaf, roll_act = [], [], [], []
            roll_cf = []
            cf_rollouts = item.get("ga_rollout_cf_actions")
            for c, (step_rollouts, action_rollouts) in enumerate(zip(
                item["ga_rollout_steps"], item["ga_rollout_actions"]
            )):
                c_tokens, c_first, c_leaf, c_act, c_cf = [], [], [], [], []
                for r, (outcomes, actions) in enumerate(
                    zip(step_rollouts, action_rollouts)
                ):
                    if cf_rollouts is not None:
                        # per depth h: catalogue indices of infeasible intents
                        # at the rollout state before rollout action h
                        c_cf.append([
                            [cat_index[tuple(a)] for a in depth_cf]
                            for depth_cf in cf_rollouts[c][r]
                        ])
                    seq = list(prefix)
                    first = None
                    # rollout outcome sequences carry the factual prefix
                    # outcomes (steps[:t]) first; the new outcomes follow.
                    new_outcomes = outcomes[t:]
                    acts = []
                    for h, (a, o) in enumerate(zip(actions, new_outcomes)):
                        seq.extend(a)
                        seq.extend(o)
                        if h == 0:
                            first = len(seq) - 1
                        acts.append(cat_index[tuple(a)])
                    c_tokens.append(seq)
                    c_first.append(first if first is not None else len(seq) - 1)
                    c_leaf.append(len(seq) - 1)
                    c_act.append(acts)
                roll_tokens.append(c_tokens)
                roll_first.append(c_first)
                roll_leaf.append(c_leaf)
                roll_act.append(c_act)
                roll_cf.append(c_cf)
            out.update(
                ga_t=t,
                ga_anchor_pos=anchor_pos,
                ga_horizon=int(item["ga_horizon"]),
                ga_cand_cat=cand_cat,
                ga_alt_invalid=alt_invalid,
                ga_roll_tokens=roll_tokens,
                ga_roll_first=roll_first,
                ga_roll_leaf=roll_leaf,
                ga_roll_act=roll_act,
                ga_roll_cf=roll_cf,
            )
        return out


def collate_flat(batch: list[dict], pad_id: int) -> dict:
    B = len(batch)
    main_len = [len(b["tokens"]) for b in batch]
    cat_n = [len(b["catalogue"]) for b in batch]
    cat_len = [sum(len(p) for p in b["catalogue"]) for b in batch]
    L = max(m + c for m, c in zip(main_len, cat_len))
    tokens = torch.full((B, L), pad_id, dtype=torch.long)
    pos_ids = torch.zeros((B, L), dtype=torch.long)
    phrase_id = torch.full((B, L), -1, dtype=torch.long)
    lm_mask = torch.zeros((B, L), dtype=torch.bool)
    T = max(len(b["a_pos"]) for b in batch)
    La = max(len(a) for b in batch for a in b["action_tokens"])
    s_pos = torch.zeros((B, T + 1), dtype=torch.long)
    a_pos = torch.zeros((B, T), dtype=torch.long)
    step_mask = torch.zeros((B, T), dtype=torch.bool)
    action_tokens = torch.full((B, T, La), pad_id, dtype=torch.long)
    Ncat = max(cat_n)
    cat_last = torch.zeros((B, Ncat), dtype=torch.long)
    cat_mask = torch.zeros((B, Ncat), dtype=torch.bool)
    anchor_pos = torch.zeros(B, dtype=torch.long)
    ga_t = torch.full((B,), -1, dtype=torch.long)
    for i, b in enumerate(batch):
        m = main_len[i]
        tokens[i, :m] = torch.tensor(b["tokens"])
        pos_ids[i, :m] = torch.arange(m)
        lm_mask[i, :m] = torch.tensor(b["lm_mask"])
        n = len(b["a_pos"])
        s_pos[i, : n + 1] = torch.tensor(b["s_pos"])
        a_pos[i, :n] = torch.tensor(b["a_pos"])
        step_mask[i, :n] = True
        for t, a in enumerate(b["action_tokens"]):
            action_tokens[i, t, : len(a)] = torch.tensor(a)
        # Catalogue block.  Without a ranking anchor the block still exists
        # (attends to the full stream end) but is unused by the losses.
        anchor = int(b.get("ga_anchor_pos", b["s_pos"][-1]))
        anchor_pos[i] = anchor
        cursor = m
        for j, phrase in enumerate(b["catalogue"]):
            k = len(phrase)
            tokens[i, cursor:cursor + k] = torch.tensor(phrase)
            pos_ids[i, cursor:cursor + k] = anchor + 1 + torch.arange(k)
            phrase_id[i, cursor:cursor + k] = j
            cat_last[i, j] = cursor + k - 1
            cat_mask[i, j] = True
            cursor += k
        if "ga_t" in b:
            ga_t[i] = b["ga_t"]
    out = {
        "tokens": tokens,
        "pos_ids": pos_ids,
        "phrase_id": phrase_id,
        "main_len": torch.tensor(main_len),
        "anchor_pos": anchor_pos,
        "lm_mask": lm_mask,
        "s_pos": s_pos,
        "a_pos": a_pos,
        "step_mask": step_mask,
        "action_tokens": action_tokens,
        "cat_last": cat_last,
        "cat_mask": cat_mask,
        "index": torch.tensor([b["index"] for b in batch]),
        "n_necessary": torch.tensor([b["n_necessary"] for b in batch]),
        "ga_t": ga_t,
    }
    ga_items = [b for b in batch if "ga_t" in b]
    if ga_items:
        C = max(len(b["ga_cand_cat"]) for b in ga_items)
        R = max(len(c) for b in ga_items for c in b["ga_roll_tokens"])
        H = max(
            (len(a) for b in ga_items for c in b["ga_roll_act"] for a in c),
            default=1,
        )
        Lr = max(
            len(s) for b in ga_items for c in b["ga_roll_tokens"] for s in c
        )
        cand_cat = torch.zeros((B, C), dtype=torch.long)
        cand_valid = torch.zeros((B, C), dtype=torch.bool)
        alt_invalid = torch.zeros((B, C - 1), dtype=torch.bool)
        roll_tokens = torch.full((B, C, R, Lr), pad_id, dtype=torch.long)
        roll_first = torch.zeros((B, C, R), dtype=torch.long)
        roll_leaf = torch.zeros((B, C, R), dtype=torch.long)
        roll_valid = torch.zeros((B, C, R), dtype=torch.bool)
        roll_act = torch.zeros((B, C, R, H), dtype=torch.long)
        roll_act_mask = torch.zeros((B, C, R, H), dtype=torch.bool)
        Kc = max(
            (len(d) for b in ga_items for c in b["ga_roll_cf"] for r in c for d in r),
            default=0,
        )
        roll_cf = torch.zeros((B, C, R, H, max(Kc, 1)), dtype=torch.long)
        roll_cf_mask = torch.zeros((B, C, R, H, max(Kc, 1)), dtype=torch.bool)
        horizon = torch.ones(B, dtype=torch.long)
        for i, b in enumerate(batch):
            if "ga_t" not in b:
                continue
            cc = b["ga_cand_cat"]
            cand_cat[i, : len(cc)] = torch.tensor(cc)
            cand_valid[i, : len(cc)] = True
            inv = b["ga_alt_invalid"]
            if inv:
                alt_invalid[i, : len(inv)] = torch.tensor(inv)
            horizon[i] = b["ga_horizon"]
            for c in range(len(cc)):
                for r, seq in enumerate(b["ga_roll_tokens"][c]):
                    roll_tokens[i, c, r, : len(seq)] = torch.tensor(seq)
                    roll_first[i, c, r] = b["ga_roll_first"][c][r]
                    roll_leaf[i, c, r] = b["ga_roll_leaf"][c][r]
                    roll_valid[i, c, r] = True
                    acts = b["ga_roll_act"][c][r]
                    if acts:
                        roll_act[i, c, r, : len(acts)] = torch.tensor(acts)
                        roll_act_mask[i, c, r, : len(acts)] = True
                    if b["ga_roll_cf"] and b["ga_roll_cf"][c]:
                        for h, depth_cf in enumerate(b["ga_roll_cf"][c][r]):
                            if depth_cf and h < H:
                                roll_cf[i, c, r, h, : len(depth_cf)] = torch.tensor(depth_cf)
                                roll_cf_mask[i, c, r, h, : len(depth_cf)] = True
        out.update(
            ga_cand_cat=cand_cat,
            ga_cand_valid=cand_valid,
            ga_alt_invalid=alt_invalid,
            ga_roll_tokens=roll_tokens,
            ga_roll_first=roll_first,
            ga_roll_leaf=roll_leaf,
            ga_roll_valid=roll_valid,
            ga_roll_act=roll_act,
            ga_roll_act_mask=roll_act_mask,
            ga_roll_cf=roll_cf,
            ga_roll_cf_mask=roll_cf_mask,
            ga_requested_horizon=horizon,
        )
    return out


def build_block_attention(
    tokens: torch.Tensor,
    pad_id: int,
    main_len: torch.Tensor,
    anchor_pos: torch.Tensor,
    phrase_id: torch.Tensor,
) -> torch.Tensor:
    """Allowed-attention matrix ``[B, L, L]`` (True = may attend).

    Main-stream queries are plain causal.  Phrase-block queries see the
    main stream up to and including ``anchor_pos`` plus the earlier tokens
    of their own phrase.  Pads are never attended; rows with no allowed key
    fall back to position 0 (their outputs are never read).
    """
    B, L = tokens.shape
    device = tokens.device
    idx = torch.arange(L, device=device)
    q = idx.view(1, L, 1)
    k = idx.view(1, 1, L)
    valid_k = (tokens != pad_id).unsqueeze(1)
    causal = k <= q
    is_main_q = (q < main_len.view(B, 1, 1))
    main_allowed = causal & (k < main_len.view(B, 1, 1))
    same_phrase = phrase_id.unsqueeze(2) == phrase_id.unsqueeze(1)
    block_allowed = (k <= anchor_pos.view(B, 1, 1)) | (same_phrase & causal)
    allowed = torch.where(is_main_q, main_allowed, block_allowed) & valid_k
    dead = ~allowed.any(-1)
    allowed[..., 0] |= dead
    return allowed
