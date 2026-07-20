"""Nested high-level subgoal CEM and low-level token planning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from textjepa.data.faithful_token_edits import _apply
from textjepa.planning.multiscale_edit_mpc import (
    BeamNode, Buffer, Edit, MultiscaleEditMPC, copy_buffer, flatten,
)


def sentence_distance(left: torch.Tensor, left_mask: torch.Tensor,
                      right: torch.Tensor, right_mask: torch.Tensor):
    valid = left_mask & right_mask
    distance = (
        F.layer_norm(left, left.shape[-1:])
        - F.layer_norm(right, right.shape[-1:])
    ).abs().mean(-1)
    return (distance * valid.to(distance.dtype)).sum(-1) / valid.sum(-1).clamp_min(1)


@dataclass
class LowLevelPlan:
    first_action: Edit | None
    residual: float
    score: float
    endpoint: Buffer


@dataclass
class HierarchicalPlan:
    first_action: Edit | None
    macro_code: torch.Tensor
    subgoal: torch.Tensor
    high_cost: float
    reachability_residual: float
    decoded_actions: tuple[Edit, ...]
    mode: str


class HierarchicalEditCEM:
    """CEM over macro codes whose predicted states become lower subgoals."""

    MODES = {"subgoal", "decoder_open_loop", "decoder", "decoder_refine"}

    def __init__(
        self, primitive: MultiscaleEditMPC, mode: str = "subgoal",
        high_horizon: int = 1, candidates: int = 32, iterations: int = 3,
        elites: int = 4, reachability_topk: int = 4,
        low_horizon: int = 4, low_goal_weight: float = 2.0,
        high_prior_weight: float = 0.05,
        reachability_weight: float = 1.0, cem_alpha: float = 0.1,
    ):
        if mode not in self.MODES:
            raise ValueError(f"unknown hierarchical planning mode: {mode}")
        if not primitive.model.use_macro:
            raise ValueError("hierarchical CEM requires a macro model")
        if mode != "subgoal" and primitive.model.macro_decoder is None:
            raise ValueError("decoder planning requires a macro decoder")
        self.primitive = primitive
        self.model = primitive.model
        self.mode = mode
        self.high_horizon = int(high_horizon)
        self.candidates = int(candidates)
        self.iterations = int(iterations)
        self.elites = int(elites)
        self.reachability_topk = int(reachability_topk)
        self.low_horizon = int(low_horizon)
        self.low_goal_weight = float(low_goal_weight)
        self.high_prior_weight = float(high_prior_weight)
        self.reachability_weight = float(reachability_weight)
        self.cem_alpha = float(cem_alpha)

    @torch.no_grad()
    def low_plan(self, prompt: Buffer, current: Buffer,
                 subgoal: torch.Tensor,
                 subgoal_mask: torch.Tensor) -> LowLevelPlan:
        beam = [BeamNode(copy_buffer(current), 0.0, (), (), (), None, None)]
        for _ in range(self.low_horizon):
            expanded = []
            for node in beam:
                expanded.extend(self.primitive.expand(
                    prompt, node, allow_refinement=False
                ))
            if not expanded:
                break
            rescored = []
            for node in expanded:
                encoded = self.primitive.encode(prompt, node.buffer)
                residual = float(sentence_distance(
                    encoded.sentences, encoded.sentence_mask,
                    subgoal, subgoal_mask,
                ))
                node.score -= self.low_goal_weight * residual
                rescored.append((node, residual))
            rescored.sort(key=lambda item: item[0].score, reverse=True)
            beam = [item[0] for item in rescored[:self.primitive.beam_width]]
        if not beam or beam[0].root is None:
            return LowLevelPlan(None, float("inf"), float("-inf"), current)
        endpoint = self.primitive.encode(prompt, beam[0].buffer)
        residual = float(sentence_distance(
            endpoint.sentences, endpoint.sentence_mask, subgoal, subgoal_mask
        ))
        return LowLevelPlan(
            beam[0].root, residual, beam[0].score, beam[0].buffer
        )

    @torch.no_grad()
    def decode_option(self, prompt: Buffer, current: Buffer,
                      macro: torch.Tensor, closed_loop: bool = True):
        buffer = copy_buffer(current)
        actions = []
        initial = self.primitive.encode(prompt, buffer)
        for step in range(self.model.macro_k):
            encoded = (
                self.primitive.encode(prompt, buffer) if closed_loop else initial
            )
            dummy = torch.zeros(1, dtype=torch.long,
                                device=encoded.tokens.device)
            position_logits, _ = self.model.macro_decoder(
                encoded.tokens, encoded.token_mask, encoded.prompt, macro,
                torch.tensor([step], device=encoded.tokens.device), dummy,
            )
            tokens = flatten(buffer)
            available = [
                index for index, token in enumerate(tokens)
                if token == self.primitive.mask_id
            ] or list(range(len(tokens)))
            allowed = torch.zeros_like(position_logits, dtype=torch.bool)
            allowed[:, available] = True
            position_logits = position_logits.masked_fill(~allowed, -torch.inf)
            position = position_logits.argmax(-1)
            _, content_logits = self.model.macro_decoder(
                encoded.tokens, encoded.token_mask, encoded.prompt, macro,
                torch.tensor([step], device=encoded.tokens.device), position,
            )
            content_logits[:, list(self.primitive.excluded)] = -torch.inf
            content_logits[:, tokens[int(position)]] = -torch.inf
            token = int(content_logits.argmax(-1))
            action = ("replace", int(position), token)
            _apply(buffer, action)
            actions.append(action)
        return tuple(actions), buffer

    @torch.no_grad()
    def _roll_high(self, start, start_mask, macros):
        states = start.expand(len(macros), -1, -1)
        mask = start_mask.expand(len(macros), -1)
        first = None
        prior_nll = states.new_zeros(len(macros))
        for level in range(self.high_horizon):
            current_global = (
                states * mask.unsqueeze(-1).to(states.dtype)
            ).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            mu, logvar = self.model.macro_model.prior_params(current_global)
            action = macros[:, level]
            prior_nll += 0.5 * (
                logvar + (action - mu).square() * (-logvar).exp()
            ).sum(-1)
            states = self.model.macro_pred(states, mask, action, None)
            if first is None:
                first = states.clone()
        global_state = (
            states * mask.unsqueeze(-1).to(states.dtype)
        ).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        terminal_distance = self.model.value_head(global_state).squeeze(-1)
        return first, states, terminal_distance, prior_nll

    @torch.no_grad()
    def first_action(self, prompt: Buffer, current: Buffer) -> HierarchicalPlan:
        encoded = self.primitive.encode(prompt, current)
        start = encoded.sentences
        mask = encoded.sentence_mask
        mu, logvar = self.model.macro_model.prior_params(encoded.global_state)
        mean = mu[:, None].expand(-1, self.high_horizon, -1).clone()
        std = (0.5 * logvar).exp()[:, None].expand_as(mean).clamp_min(0.05)
        best = None
        elite_n = min(self.elites, self.candidates)
        for _ in range(self.iterations):
            raw = mean + std * torch.randn(
                self.candidates, self.high_horizon, mean.shape[-1],
                device=mean.device,
            )
            first, _, terminal, prior_nll = self._roll_high(start, mask, raw)
            cost = terminal + self.high_prior_weight * prior_nll
            retained = cost.topk(
                min(max(self.reachability_topk, elite_n), self.candidates),
                largest=False,
            ).indices
            reach = torch.full_like(cost, torch.inf)
            plans = {}
            for index in retained.tolist():
                subgoal = first[index:index + 1]
                if self.mode == "subgoal":
                    plan = self.low_plan(prompt, current, subgoal, mask)
                    plans[index] = plan
                    reach[index] = plan.residual
                else:
                    decoded, endpoint_buffer = self.decode_option(
                        prompt, current, raw[index:index + 1, 0],
                        closed_loop=self.mode != "decoder_open_loop",
                    )
                    endpoint = self.primitive.encode(prompt, endpoint_buffer)
                    reach[index] = sentence_distance(
                        endpoint.sentences, endpoint.sentence_mask,
                        subgoal, mask,
                    )
                    plans[index] = (decoded, endpoint_buffer)
            refined = cost + self.reachability_weight * reach
            ids = refined.topk(elite_n, largest=False).indices
            elite = raw[ids]
            new_mean = elite.mean(0, keepdim=True)
            new_std = elite.std(0, unbiased=False, keepdim=True).clamp_min(0.05)
            mean = self.cem_alpha * mean + (1 - self.cem_alpha) * new_mean
            std = self.cem_alpha * std + (1 - self.cem_alpha) * new_std
            index = int(refined.argmin())
            if best is None or float(refined[index]) < best[0]:
                best = (
                    float(refined[index]), raw[index:index + 1].clone(),
                    first[index:index + 1].clone(), float(reach[index]),
                    plans[index],
                )
        assert best is not None
        high_cost, macro, subgoal, residual, plan = best
        decoded_actions: tuple[Edit, ...] = ()
        if self.mode == "subgoal":
            first_action = plan.first_action
        else:
            decoded_actions, _ = plan
            if self.mode == "decoder_refine":
                refined = self.low_plan(prompt, current, subgoal, mask)
                first_action = refined.first_action
                residual = refined.residual
            else:
                first_action = decoded_actions[0] if decoded_actions else None
        return HierarchicalPlan(
            first_action, macro[:, 0], subgoal, high_cost, residual,
            decoded_actions, self.mode,
        )
