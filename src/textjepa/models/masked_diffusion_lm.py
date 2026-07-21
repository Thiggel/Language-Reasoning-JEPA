"""Absorbing-mask discrete diffusion language model for edit controls."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from textjepa.models.layers import encoder_stack


def select_terminal_buffers(buffer_tokens: torch.Tensor,
                            step_mask: torch.Tensor) -> torch.Tensor:
    """Select each example's clean T+1 state from a padded trajectory batch."""
    terminal = step_mask.long().sum(-1)
    if terminal.ge(buffer_tokens.shape[1]).any():
        raise ValueError("step mask points beyond the available buffer states")
    row = torch.arange(len(buffer_tokens), device=buffer_tokens.device)
    return buffer_tokens[row, terminal]


class MaskedDiffusionLM(nn.Module):
    """MDLM-style prompt-conditional model with SUBS parameterization."""

    def __init__(self, vocab_size: int, pad_id: int, mask_id: int,
                 d_model: int = 128, n_layers: int = 4, n_heads: int = 8,
                 ff_mult: int = 4, max_sequence_len: int = 2048,
                 dropout: float = 0.0):
        super().__init__()
        if dropout != 0:
            raise ValueError("diffusion comparison requires dropout=0")
        self.vocab_size = int(vocab_size)
        self.pad_id = int(pad_id)
        self.mask_id = int(mask_id)
        self.max_sequence_len = int(max_sequence_len)
        self.token = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position = nn.Parameter(torch.zeros(1, max_sequence_len, d_model))
        self.segment = nn.Parameter(torch.zeros(2, d_model))
        self.encoder = encoder_stack(
            d_model, n_layers, n_heads, ff_mult, dropout
        )
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.token.weight
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.segment, std=0.02)

    def pack_clean(self, prompt: torch.Tensor, buffer: torch.Tensor):
        """Pack valid ``[prompt | response]`` tokens and retain ownership."""
        rows, response_flags = [], []
        for row in range(len(prompt)):
            p = prompt[row].reshape(-1)
            p = p[p.ne(self.pad_id)]
            r = buffer[row].reshape(-1)
            r = r[r.ne(self.pad_id)]
            rows.append(torch.cat([p, r]))
            response_flags.append(torch.cat([
                torch.zeros_like(p, dtype=torch.bool),
                torch.ones_like(r, dtype=torch.bool),
            ]))
        width = max(max(x.numel(), 1) for x in rows)
        if width > self.max_sequence_len:
            raise ValueError(
                f"packed length {width} exceeds max_sequence_len="
                f"{self.max_sequence_len}"
            )
        clean = prompt.new_full((len(rows), width), self.pad_id)
        valid = torch.zeros(len(rows), width, dtype=torch.bool,
                            device=prompt.device)
        response = torch.zeros_like(valid)
        for row, (tokens, flags) in enumerate(zip(rows, response_flags)):
            clean[row, :tokens.numel()] = tokens
            valid[row, :tokens.numel()] = True
            response[row, :flags.numel()] = flags
        return clean, valid, response

    def logits(self, tokens: torch.Tensor, valid: torch.Tensor,
               response: torch.Tensor):
        h = (self.token(tokens) + self.position[:, :tokens.shape[1]]
             + self.segment[response.long()])
        key_pad = ~valid
        key_pad = key_pad.clone()
        key_pad[key_pad.all(-1), 0] = False
        h = self.norm(self.encoder(h, src_key_padding_mask=key_pad))
        return self.output(h), h

    def corrupt(self, clean: torch.Tensor, response: torch.Tensor,
                noise: torch.Tensor, *, random: torch.Tensor | None = None):
        """Sample the absorbing forward process q(x_t | x_0)."""
        if noise.ndim == 1:
            noise = noise[:, None]
        draw = torch.rand(clean.shape, device=clean.device) if random is None else random
        masked = response & draw.lt(noise)
        return clean.masked_fill(masked, self.mask_id), masked

    def mdlm_loss(self, clean: torch.Tensor, valid: torch.Tensor,
                  response: torch.Tensor, noise: torch.Tensor | None = None):
        """Rao--Blackwellized continuous-time ELBO for alpha(t)=1-t.

        Normalization is by all response tokens. Averaging only selected masks
        and then applying 1/t would introduce an erroneous second reweighting.
        """
        if noise is None:
            noise = torch.rand(len(clean), device=clean.device)
        noise = noise.clamp_min(1e-4)
        noised, masked = self.corrupt(clean, response, noise)
        logits, states = self.logits(noised, valid, response)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), clean.reshape(-1),
            reduction="none",
        ).reshape_as(clean)
        weight = masked.to(ce.dtype) / noise[:, None]
        loss = (ce * weight).sum() / response.sum().clamp_min(1)
        return loss, {
            "clean": clean, "noised": noised, "valid": valid,
            "response": response, "masked": masked, "noise": noise,
            "logits": logits, "states": states,
        }

    @torch.no_grad()
    def sample(self, prompt: torch.Tensor, shape_buffer: torch.Tensor,
               steps: int | None = None, temperature: float = 1.0,
               stochastic: bool = False, schedule: str = "confidence"):
        """Reveal response tokens without remasking.

        ``confidence`` performs MaskGIT-style coordinate selection: every
        network evaluation commits exactly one token per unfinished example,
        choosing the unresolved position whose argmax token has the highest
        softmax probability.  ``random`` retains the historical SUBS schedule
        for explicit backward-compatible diagnostics.
        """
        clean_shape, valid, response = self.pack_clean(prompt, shape_buffer)
        tokens = clean_shape.masked_fill(response, self.mask_id)
        if schedule not in {"confidence", "random"}:
            raise ValueError(f"unknown unmasking schedule: {schedule}")
        if steps is None:
            steps = int(response.sum(-1).max().item())
        if steps < 1:
            raise ValueError("sampling steps must be positive")
        for step in range(steps, 0, -1):
            unresolved = response & tokens.eq(self.mask_id)
            if not unresolved.any():
                break
            logits, _ = self.logits(tokens, valid, response)
            logits[..., self.mask_id] = -torch.inf
            logits[..., self.pad_id] = -torch.inf
            if stochastic:
                proposal = torch.distributions.Categorical(
                    logits=logits / max(float(temperature), 1e-4)
                ).sample()
            else:
                proposal = logits.argmax(-1)
            if schedule == "confidence":
                confidence = logits.softmax(-1).amax(-1)
                confidence = confidence.masked_fill(~unresolved, -torch.inf)
                position = confidence.argmax(-1)
                active = unresolved.any(-1)
                reveal = torch.zeros_like(unresolved)
                row = torch.arange(len(tokens), device=tokens.device)[active]
                reveal[row, position[active]] = True
            else:
                reveal = unresolved & torch.rand_like(tokens.float()).lt(
                    1.0 / step
                )
                if step == 1:
                    reveal = unresolved
            tokens = torch.where(reveal, proposal, tokens)
        return tokens, valid, response
