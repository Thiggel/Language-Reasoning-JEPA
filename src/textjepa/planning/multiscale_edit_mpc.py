"""Target-free receding-horizon generation for multiscale edit JEPA.

Replacement mechanics are known, so hypothetical token buffers are updated
exactly.  The learned JEPA predicts their latent consequences; a goal-distance
value and V(s,a), both distilled during training, rank plans without exposing
the clean target.  Hierarchical models additionally rerank every executable
K-token span with their macro prior and macro transition/value prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from textjepa.data.faithful_token_edits import MASK_TOKEN, OPS, _apply


Buffer = list[list[int]]
Edit = tuple[str, int, int | None]


def copy_buffer(buffer: Buffer) -> Buffer:
    return [list(sentence) for sentence in buffer]


def flatten(buffer: Buffer) -> list[int]:
    return [token for sentence in buffer for token in sentence]


def _chunks_tensor(chunks: Buffer, pad_id: int, device: torch.device):
    count = max(len(chunks), 1)
    width = max((len(chunk) for chunk in chunks), default=1)
    value = torch.full(
        (1, count, max(width, 1)), pad_id, dtype=torch.long, device=device
    )
    for index, chunk in enumerate(chunks):
        value[0, index, :len(chunk)] = torch.tensor(chunk, device=device)
    return value


@dataclass
class EncodedState:
    tokens: torch.Tensor
    token_mask: torch.Tensor
    sentence_ids: torch.Tensor
    sentences: torch.Tensor
    sentence_mask: torch.Tensor
    global_state: torch.Tensor
    prompt: torch.Tensor


@dataclass
class BeamNode:
    buffer: Buffer
    score: float
    actions: tuple[Edit, ...]
    action_codes: tuple[torch.Tensor, ...]
    encoded_history: tuple[EncodedState, ...]
    root: Edit | None
    root_q: float | None


class MultiscaleEditMPC:
    """Executable discrete beam MPC; execute one action, then replan."""

    def __init__(
        self, model, vocab, device: str = "cuda:0", beam_width: int = 8,
        top_positions: int = 4, top_tokens: int = 4,
        max_candidates: int = 16, prior_weight: float = 0.05,
        action_value_weight: float = 1.0, state_value_weight: float = 0.25,
        macro_prior_weight: float = 0.05,
        macro_value_weight: float = 0.25,
        use_base_prior: bool = True,
    ):
        if use_base_prior and model.base_prior is None:
            raise ValueError("MPC requires a checkpoint trained with base_prior")
        self.model = model
        self.vocab = vocab
        self.device = torch.device(device)
        self.beam_width = int(beam_width)
        self.top_positions = int(top_positions)
        self.top_tokens = int(top_tokens)
        self.max_candidates = int(max_candidates)
        self.prior_weight = float(prior_weight)
        self.action_value_weight = float(action_value_weight)
        self.state_value_weight = float(state_value_weight)
        self.macro_prior_weight = float(macro_prior_weight)
        self.macro_value_weight = float(macro_value_weight)
        self.use_base_prior = bool(use_base_prior)
        self.mask_id = vocab.token_to_id[MASK_TOKEN]
        self.excluded = {
            vocab.pad_id, vocab.token_to_id[vocab.UNK], self.mask_id,
        }

    @torch.no_grad()
    def encode(self, prompt: Buffer, buffer: Buffer) -> EncodedState:
        p = _chunks_tensor(prompt, self.vocab.pad_id, self.device)
        x = _chunks_tensor(buffer, self.vocab.pad_id, self.device)
        token, token_mask, ids, sentence, sentence_mask, _ = self.model.encoder(p, x)
        prompt_mask = p.ne(self.vocab.pad_id)
        prompt_emb = (
            self.model.encoder.tok(p).reshape(1, -1, token.shape[-1])
            * prompt_mask.reshape(1, -1, 1)
        ).sum(1) / prompt_mask.reshape(1, -1).sum(1, keepdim=True).clamp_min(1)
        if self.model.use_sentence:
            weight = sentence_mask.unsqueeze(-1).to(sentence.dtype)
            global_state = (sentence * weight).sum(1) / weight.sum(1).clamp_min(1)
        else:
            weight = token_mask.unsqueeze(-1).to(token.dtype)
            global_state = (token * weight).sum(1) / weight.sum(1).clamp_min(1)
        return EncodedState(
            token, token_mask, ids, sentence, sentence_mask,
            global_state, prompt_emb,
        )

    @torch.no_grad()
    def candidates(self, state: EncodedState, prompt: Buffer, buffer: Buffer,
                   allow_refinement: bool) -> list[tuple[Edit, float]]:
        current = flatten(buffer)
        available = [i for i, token in enumerate(current) if token == self.mask_id]
        if not available:
            if not allow_refinement:
                return []
            available = list(range(len(current)))
        if not self.use_base_prior:
            # Information-matched flat control: bounded deterministic-uniform
            # proposals from observable prompt/current tokens only.
            offset = sum(current) % len(available)
            positions = (available[offset:] + available[:offset])[
                :self.top_positions
            ]
            token_pool = sorted(set(flatten(prompt) + current) - self.excluded)
            if not token_pool:
                return []
            token_offset = sum(current) % len(token_pool)
            token_pool = token_pool[token_offset:] + token_pool[:token_offset]
            proposals = [
                (("replace", position, token),
                 -math.log(len(available)) - math.log(len(token_pool)))
                for position in positions
                for token in token_pool[:self.top_tokens]
                if token != current[position]
            ]
            return proposals[:self.max_candidates]
        dummy = torch.zeros(1, dtype=torch.long, device=self.device)
        position_logits, _ = self.model.replacement_prior(
            state.tokens, state.token_mask, state.prompt, dummy
        )
        allowed = torch.zeros_like(position_logits, dtype=torch.bool)
        allowed[:, available] = True
        position_logp = position_logits.masked_fill(~allowed, -torch.inf).log_softmax(-1)
        count = min(self.top_positions, len(available))
        positions = position_logp.topk(count, -1).indices[0]
        _, token_logits = self.model.replacement_prior(
            state.tokens.expand(count, -1, -1),
            state.token_mask.expand(count, -1),
            state.prompt.expand(count, -1), positions,
        )
        token_logits[:, list(self.excluded)] = -torch.inf
        token_logp = token_logits.log_softmax(-1)
        proposals: list[tuple[Edit, float]] = []
        for row, position in enumerate(positions.tolist()):
            accepted = 0
            for token in token_logp[row].argsort(descending=True).tolist():
                if token == current[position]:
                    continue
                proposals.append((
                    ("replace", position, token),
                    float(position_logp[0, position] + token_logp[row, token]),
                ))
                accepted += 1
                if accepted >= self.top_tokens:
                    break
        proposals.sort(key=lambda item: item[1], reverse=True)
        return proposals[:self.max_candidates]

    @torch.no_grad()
    def _action_codes(self, state: EncodedState,
                      actions: list[Edit]) -> torch.Tensor:
        count = len(actions)
        operations = torch.full(
            (count,), OPS["replace"], dtype=torch.long, device=self.device
        )
        positions = torch.tensor([a[1] for a in actions], device=self.device)
        content = self.model.encoder.tok(torch.tensor(
            [int(a[2]) for a in actions], device=self.device
        ))
        module = (
            self.model.sentence_action if self.model.token_pred is None
            else self.model.token_pred.encode_action
        )
        return module(
            state.tokens.expand(count, -1, -1),
            state.token_mask.expand(count, -1), operations, positions, content,
        )

    @torch.no_grad()
    def _predicted_next_distance(self, state: EncodedState,
                                 actions: list[Edit],
                                 codes: torch.Tensor) -> torch.Tensor:
        count = len(actions)
        operations = torch.full(
            (count,), OPS["replace"], dtype=torch.long, device=self.device
        )
        positions = torch.tensor([a[1] for a in actions], device=self.device)
        content = self.model.encoder.tok(torch.tensor(
            [int(a[2]) for a in actions], device=self.device
        ))
        if self.model.token_pred is not None:
            token_pred, predicted_mask = self.model.token_pred(
                state.tokens.expand(count, -1, -1),
                state.token_mask.expand(count, -1), operations, positions,
                content, state.prompt.expand(count, -1),
            )
        if not self.model.use_sentence:
            weight = predicted_mask.unsqueeze(-1).to(token_pred.dtype)
            predicted = (token_pred * weight).sum(1) / weight.sum(1).clamp_min(1)
            return self.model.value_head(predicted).squeeze(-1)
        affected = self.model.affected_sentences(
            state.sentence_ids.expand(count, -1),
            state.token_mask.expand(count, -1), operations, positions,
        )
        if self.model.variant in {"sentence", "sentence_macro"}:
            base = state.sentences.expand(count, -1, -1)
            current = None
        else:
            next_ids, next_mask, _ = self.model.transition_sentence_ids(
                state.sentence_ids.expand(count, -1),
                state.token_mask.expand(count, -1), operations, positions,
            )
            base, _, _ = self.model.encoder.pool_sentences(
                token_pred, predicted_mask & next_mask, next_ids,
                state.sentences.shape[1],
            )
            current = state.sentences.expand(count, -1, -1)
        predicted_sentence = self.model.sentence_pred(
            base, state.sentence_mask.expand(count, -1), codes, affected,
            current,
        )
        weight = state.sentence_mask.expand(count, -1).unsqueeze(-1).to(
            predicted_sentence.dtype
        )
        predicted = (
            (predicted_sentence * weight).sum(1)
            / weight.sum(1).clamp_min(1)
        )
        return self.model.value_head(predicted).squeeze(-1)

    @torch.no_grad()
    def expand(self, prompt: Buffer, node: BeamNode,
               allow_refinement: bool) -> list[BeamNode]:
        encoded = self.encode(prompt, node.buffer)
        proposed = self.candidates(
            encoded, prompt, node.buffer, allow_refinement
        )
        if not proposed:
            return []
        actions = [action for action, _ in proposed]
        codes = self._action_codes(encoded, actions)
        q = self.model.action_value(
            encoded.global_state.expand(len(actions), -1), codes
        )
        predicted_distance = self._predicted_next_distance(encoded, actions, codes)
        children = []
        for index, ((action, log_prior), action_q) in enumerate(zip(proposed, q)):
            outcome = copy_buffer(node.buffer)
            _apply(outcome, action)
            root = node.root if node.root is not None else action
            root_q = node.root_q if node.root_q is not None else float(action_q)
            child_codes = (*node.action_codes, codes[index].detach())
            history = (*node.encoded_history, encoded)
            score = (
                node.score + self.prior_weight * log_prior
                + self.action_value_weight * float(action_q)
                - self.state_value_weight * float(predicted_distance[index])
            )
            if self.model.use_macro and len(child_codes) % self.model.macro_k == 0:
                k = self.model.macro_k
                macro = self.model.macro_model(torch.stack(child_codes[-k:])[None])
                start = history[-k]
                mu, logvar = self.model.macro_model.prior_params(start.global_state)
                nll = 0.5 * (
                    logvar + (macro - mu).square() * (-logvar).exp()
                ).sum(-1)
                macro_sentence = self.model.macro_pred(
                    start.sentences, start.sentence_mask, macro, None,
                )
                weight = start.sentence_mask.unsqueeze(-1).to(macro_sentence.dtype)
                macro_global = (
                    (macro_sentence * weight).sum(1)
                    / weight.sum(1).clamp_min(1)
                )
                macro_distance = self.model.value_head(macro_global).squeeze(-1)
                score += (
                    -self.macro_prior_weight * float(nll)
                    -self.macro_value_weight * float(macro_distance)
                )
            children.append(BeamNode(
                outcome, score, (*node.actions, action), child_codes,
                history, root, root_q,
            ))
        return children

    @torch.no_grad()
    def first_action(self, prompt: Buffer, current: Buffer, horizon: int,
                     allow_refinement: bool = False):
        beam = [BeamNode(copy_buffer(current), 0.0, (), (), (), None, None)]
        for _ in range(int(horizon)):
            expanded = []
            for node in beam:
                expanded.extend(self.expand(prompt, node, allow_refinement))
            if not expanded:
                break
            expanded.sort(key=lambda item: item.score, reverse=True)
            beam = expanded[:self.beam_width]
        if not beam or beam[0].root is None:
            return None, {}, float("-inf")
        root_scores: dict[Edit, float] = {}
        for node in beam:
            root_scores[node.root] = max(
                root_scores.get(node.root, -math.inf), node.score
            )
        return beam[0].root, root_scores, float(beam[0].root_q)
