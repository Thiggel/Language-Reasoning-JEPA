"""Flat-backbone intent JEPA.

One causal token transformer (the token-LM architecture, optionally
initialized from a trained token LM) reads the trace stream
``prompt, intent_1, outcome_1, ...``.

* state  s_t   = hidden state at the last token of outcome_t (s_0: end of
                 the prompt);
* action a_t   = hidden state at the last token of intent_t, i.e. the intent
                 phrase encoded IN CONTEXT (candidates are encoded the same
                 way through a phrase block that attends to the prefix);
* predictor    F(s_t, a_t) -> s_{t+1}, trained toward the EMA teacher's
                 LN-normalized s_{t+1};
* Energy       horizon-blind endpoint head E(root, imagined endpoint, s_0)
                 trained by ranking recursively imagined rollouts against the
                 EMA-latent distance of the TRUE rollout outcomes to the
                 encoded solved state (+ 0.25 pair-difference regression),
                 and by the self-supervised feasibility contrast
                 softplus(E(F(s,u_obs)) - E(F(s,u_cf))) over counterfactual
                 intents (label = which continuation actually occurred);
* LDAD         decode the observed intent tokens from s_{t+1} - s_t;
* intent prior next-token CE on the stream (intent and, by default, outcome
                 tokens) -- the generative proposer for menu-free planning and
                 the replacement of the old ``chunk_pred`` anti-collusion term.

The teacher is an EMA copy of the encoder.  No symbolic quantity is read
anywhere in training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from textjepa.data.flat_stream import build_block_attention
from textjepa.models.delta_decoder import ObservedActionDecoder
from textjepa.models.ema import EMATeacher
from textjepa.models.heads import HorizonEnergyHead, QuasimetricEnergyHead
from textjepa.models.lm_baseline import DecoderLM
from textjepa.models.outputs import JEPAOutputs
from textjepa.models.predictor import (
    ActionConditionedPredictor,
    CausalHistoryPredictor,
)


def _ln(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, x.shape[-1:])


class FlatIntentJEPA(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        ff_mult: int = 4,
        max_len: int = 4096,
        pos_kind: str = "learned",
        init_from_lm: str | None = None,
        encoder_mode: str = "full",  # full | frozen | lora
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        predictor_kind: str = "mlp",  # mlp | causal
        predictor_hidden_mult: int = 4,
        predictor_layers: int = 2,
        predictor_residual: bool = True,
        predictor_heads: int = 8,
        energy_hidden_mult: int = 2,
        geo_horizon_input: bool = False,
        observed_action_ldad: bool = True,
        ldad_max_len: int = 16,
        energy_cf_feasibility_rank: bool = True,
        energy_cf_scope: str = "all",  # all | invalid_only
        energy_prefix_rank: bool = False,
        energy_imagined_rank: bool = False,
        energy_imagined_depth: int = 4,
        energy_imagined_kcat: int = 8,
        energy_prefix_cf_kind: str = "all",     # all | premature | resolved
        energy_prefix_depth_bias: str = "uniform",  # uniform | late
        energy_prefix_n_insert: int = 1,
        energy_head_kind: str = "mlp",  # mlp | quasimetric
        energy_monotone: bool = False,
        hindsight_long_horizon_rank: bool = False,
        mismatched_goal_rank: bool = False,
        value_detach: bool = False,
        teacher_chunk: int = 96,
        lm_loss_on: str = "all_solution",
        lm_detach_state: bool = False,
        latent_rollout_ks: tuple[int, ...] = (),
        # Accepted for checkpoint-config compatibility with the FSA training
        # model (patches/fsa_flat_intent_jepa.py): the block-encoded rollout
        # teacher pass is a training-time device only; planning and probing
        # never use it, so both keys are ignored here.
        rollout_blocks: bool = False,
        rollout_block_chunk: int = 4,
    ):
        super().__init__()
        if encoder_mode not in {"full", "frozen", "lora"}:
            raise ValueError(f"unknown encoder_mode: {encoder_mode}")
        if predictor_kind not in {"mlp", "causal"}:
            raise ValueError(f"unknown predictor_kind: {predictor_kind}")
        if energy_cf_scope not in {"all", "invalid_only"}:
            raise ValueError(f"unknown energy_cf_scope: {energy_cf_scope}")
        if lm_loss_on not in {"intent", "all_solution"}:
            raise ValueError(f"unknown lm_loss_on: {lm_loss_on}")
        self.pad_id = pad_id
        self.d_model = d_model
        self.encoder_mode = encoder_mode
        self.predictor_kind = predictor_kind
        self.energy_cf_feasibility_rank = bool(energy_cf_feasibility_rank)
        self.energy_prefix_rank = bool(energy_prefix_rank)
        self.energy_imagined_rank = bool(energy_imagined_rank)
        self.energy_imagined_depth = int(energy_imagined_depth)
        self.energy_imagined_kcat = int(energy_imagined_kcat)
        if energy_prefix_cf_kind not in {"all", "premature", "resolved"}:
            raise ValueError(
                f"unknown energy_prefix_cf_kind: {energy_prefix_cf_kind}"
            )
        if energy_prefix_depth_bias not in {"uniform", "late"}:
            raise ValueError(
                f"unknown energy_prefix_depth_bias: {energy_prefix_depth_bias}"
            )
        self.energy_prefix_cf_kind = energy_prefix_cf_kind
        self.energy_prefix_depth_bias = energy_prefix_depth_bias
        self.energy_prefix_n_insert = max(1, int(energy_prefix_n_insert))
        self.energy_cf_scope = energy_cf_scope
        self.energy_monotone = bool(energy_monotone)
        self.hindsight_long_horizon_rank = bool(hindsight_long_horizon_rank)
        self.mismatched_goal_rank = bool(mismatched_goal_rank)
        self.value_detach = bool(value_detach)
        self.teacher_chunk = int(teacher_chunk)
        self.lm_loss_on = lm_loss_on
        self.lm_detach_state = bool(lm_detach_state)
        self.latent_rollout_ks = tuple(sorted({int(k) for k in latent_rollout_ks}))
        if self.latent_rollout_ks and min(self.latent_rollout_ks) < 1:
            raise ValueError("latent_rollout_ks must be >= 1")
        self.init_from_lm = init_from_lm
        if pos_kind not in {"learned", "rope"}:
            raise ValueError(f"unknown pos_kind: {pos_kind}")
        self.pos_kind = pos_kind
        self.encoder = DecoderLM(
            vocab_size=vocab_size, pad_id=pad_id, d_model=d_model,
            n_layers=n_layers, n_heads=n_heads, ff_mult=ff_mult,
            max_len=max_len, untie_head=self.lm_detach_state,
            pos_kind=pos_kind,
        )
        if init_from_lm:
            ckpt = torch.load(init_from_lm, map_location="cpu", weights_only=False)
            missing, unexpected = self.encoder.load_state_dict(
                ckpt["model"], strict=True
            )
            print(f"FlatIntentJEPA: encoder initialized from {init_from_lm}")
        if encoder_mode == "frozen":
            self.encoder.requires_grad_(False)
        elif encoder_mode == "lora":
            from textjepa.models.lora_backbone import LoRAConfig, attach_lora

            # nn.MultiheadAttention keeps q/k/v in one packed parameter (not
            # an nn.Linear); adapt the attention output projection and the
            # feed-forward projections instead.
            adapted = attach_lora(
                self.encoder,
                LoRAConfig(rank=lora_rank, alpha=lora_alpha,
                           targets=("out_proj", "linear1", "linear2")),
            )
            print(f"FlatIntentJEPA: LoRA on {len(adapted)} projections")
        self.teacher = EMATeacher(self.encoder)
        if predictor_kind == "mlp":
            self.predictor = ActionConditionedPredictor(
                d_model, d_model, hidden_mult=predictor_hidden_mult,
                n_hidden_layers=predictor_layers, residual=predictor_residual,
            )
        else:
            self.predictor = CausalHistoryPredictor(
                d_model, d_model, n_layers=predictor_layers,
                n_heads=predictor_heads, residual=predictor_residual,
                pos_kind=pos_kind,
            )
        if energy_head_kind not in {"mlp", "quasimetric"}:
            raise ValueError(f"unknown energy_head_kind: {energy_head_kind}")
        self.energy_head_kind = energy_head_kind
        if energy_head_kind == "quasimetric":
            # structured state potential: d_q(endpoint, G(initial)); root and
            # horizon are ignored on purpose (one ruler for every depth)
            self.horizon_energy_head = QuasimetricEnergyHead(
                d_model, hidden_mult=energy_hidden_mult,
            )
        else:
            self.horizon_energy_head = HorizonEnergyHead(
                d_model, hidden_mult=energy_hidden_mult,
                use_horizon=geo_horizon_input,
            )
        self.observed_action_decoder = (
            ObservedActionDecoder(d_model, vocab_size, ldad_max_len)
            if observed_action_ldad else None
        )
        self.observed_action_ldad_horizon = 1

    # ------------------------------------------------------------------ enc
    def _run_encoder(
        self, enc: DecoderLM, tokens: torch.Tensor,
        allowed: torch.Tensor | None = None,
        pos_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Hidden states [B, L, D] under an arbitrary allowed-attention mask."""
        B, L = tokens.shape
        x = enc.tok(tokens)
        if enc.pos is not None:
            if pos_ids is None:
                x = x + enc.pos[:, :L]
            else:
                x = x + enc.pos[0][pos_ids]
        if allowed is None:
            from textjepa.models.layers import build_causal_attention_mask

            mask = build_causal_attention_mask(tokens != enc.pad_id, enc.n_heads)
        else:
            mask = (~allowed).repeat_interleave(enc.n_heads, dim=0)
        if enc.pos is None:
            x = enc.blocks(x, mask=mask, pos_ids=pos_ids)
        else:
            x = enc.blocks(x, mask=mask)
        return enc.norm(x)

    def encode(self, tokens, allowed=None, pos_ids=None, teacher=False):
        if teacher:
            with torch.no_grad():
                return self._run_encoder(self.teacher.module, tokens, allowed, pos_ids)
        return self._run_encoder(self.encoder, tokens, allowed, pos_ids)

    @torch.no_grad()
    def teacher_gather(
        self, tokens: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Plain causal teacher pass over padded sequences ``[N, L]``,
        chunked; returns hidden states at ``positions`` ``[N, P]`` ->
        ``[N, P, D]`` without materializing every hidden state."""
        outs = []
        for start in range(0, tokens.shape[0], self.teacher_chunk):
            chunk = tokens[start:start + self.teacher_chunk]
            pos = positions[start:start + self.teacher_chunk]
            keep = int((chunk != self.pad_id).any(0).nonzero().max().item()) + 1
            h = self._run_encoder(self.teacher.module, chunk[:, :keep])
            rows = torch.arange(h.shape[0], device=h.device).unsqueeze(1)
            outs.append(h[rows, pos.clamp(max=keep - 1)])
        return torch.cat(outs, 0)

    def update_teachers(self, momentum: float) -> None:
        self.teacher.update(self.encoder, momentum)

    # ------------------------------------------------------------ dynamics
    def predict(self, states: torch.Tensor, actions: torch.Tensor,
                state_history=None, action_history=None) -> torch.Tensor:
        """One-step prediction. MLP: Markov in (state, action).  Causal: the
        observed prefix history is supplied and the action appended."""
        if self.predictor_kind == "mlp":
            return self.predictor(states, actions)
        return self.predictor.rollout(
            states, actions.unsqueeze(1), state_history, action_history
        )[:, 0]

    def imagine(self, root: torch.Tensor, action_codes: torch.Tensor,
                action_mask: torch.Tensor, state_history=None,
                action_history=None, return_prefixes: bool = False):
        """Recursive rollout; masked actions are absorbing no-ops.
        root [N, D], action_codes [N, H, D], action_mask [N, H] -> endpoint
        [N, D] (and, optionally, every prefix state [N, H+1, D] where
        prefix 0 is the root)."""
        H = action_codes.shape[1]
        prefixes = [root]
        if self.predictor_kind == "mlp":
            endpoint = root
            for h in range(H):
                proposed = self.predictor(endpoint, action_codes[:, h])
                endpoint = torch.where(
                    action_mask[:, h].unsqueeze(-1), proposed, endpoint
                )
                prefixes.append(endpoint)
        else:
            raw = self.predictor.rollout(
                root, action_codes, state_history, action_history
            )
            endpoint = root
            for h in range(H):
                endpoint = torch.where(
                    action_mask[:, h].unsqueeze(-1), raw[:, h], endpoint
                )
                prefixes.append(endpoint)
        if return_prefixes:
            return endpoint, torch.stack(prefixes, 1)
        return endpoint

    def energy(self, root, endpoint, initial, horizon=1):
        return self.horizon_energy_head(root, endpoint, initial, horizon)

    # ------------------------------------------------------------- forward
    def forward(self, batch: dict) -> JEPAOutputs:
        tokens = batch["tokens"]
        B, L = tokens.shape
        device = tokens.device
        bidx = torch.arange(B, device=device)
        allowed = build_block_attention(
            tokens, self.pad_id, batch["main_len"], batch["anchor_pos"],
            batch["phrase_id"],
        )
        H = self.encode(tokens, allowed, batch["pos_ids"])
        Ht = self.encode(tokens, allowed, batch["pos_ids"], teacher=True)
        s_pos, a_pos = batch["s_pos"], batch["a_pos"]
        step_mask = batch["step_mask"]
        s_all = H[bidx.unsqueeze(1), s_pos]  # [B, T+1, D]
        s_all_t = Ht[bidx.unsqueeze(1), s_pos]
        s0, step_states, prev_states = s_all[:, 0], s_all[:, 1:], s_all[:, :-1]
        s0_t, step_states_t = s_all_t[:, 0], s_all_t[:, 1:]
        actions = H[bidx.unsqueeze(1), a_pos]  # [B, T, D]
        cat_vecs = H[bidx.unsqueeze(1), batch["cat_last"]]  # [B, Ncat, D]
        T = step_mask.shape[1]
        if self.predictor_kind == "mlp":
            preds = self.predictor(prev_states, actions)
        else:
            preds = self.predictor(prev_states, actions, valid=step_mask)
        extras: dict = {}
        # intent prior / LM head over the main stream only
        Lm = int(batch["main_len"].max().item())
        h_lm = H[:, :Lm]
        if self.lm_detach_state:
            # Stop-gradient: the proposal head still learns to write intent
            # phrases, but ``intent_prior_lm`` can no longer pull the encoder
            # toward retaining surface detail.  The head is untied in this
            # mode (see DecoderLM.untie_head), so the CE term reaches NO
            # encoder parameter at all -- not even the token embedding.
            h_lm = h_lm.detach()
        extras["lm_logits"] = self.encoder.head(h_lm)
        extras["lm_mask"] = batch["lm_mask"][:, :Lm]
        extras["lm_tokens"] = tokens[:, :Lm]
        if self.observed_action_decoder is not None:
            extras["observed_action_logits"] = self.observed_action_decoder(
                step_states - prev_states
            )
        if self.latent_rollout_ks:
            self._latent_rollout(
                extras, prev_states, actions, step_states_t, step_mask
            )
        if self.energy_monotone:
            # cross-time Energy trace from ONE fixed root: [B, T]
            root_m = s0.unsqueeze(1).expand(-1, T, -1)
            hz = torch.arange(
                1, T + 1, device=device, dtype=s0.dtype
            ).view(1, T).expand(B, T)
            extras["energy_monotone"] = self.energy(
                root_m, step_states, s0, hz
            )
        if self.mismatched_goal_rank and B > 1:
            # GOAL-READING contrast: identical (root, endpoint) pairs, only
            # the goal/initial slot differs -- the own prompt state vs.
            # another problem's (batch rolled by one).  The paired ranking
            # loss (``MismatchedGoalRank``) forces the Energy head to READ
            # the goal input instead of scoring progress-shaped states
            # goal-blind.  Real encoded transitions at EVERY depth.
            s0_mis = torch.roll(s0, 1, dims=0)
            g_prev, g_step = prev_states, step_states
            g_own, g_mis = s0, s0_mis
            if self.value_detach:
                g_prev, g_step = prev_states.detach(), step_states.detach()
                g_own, g_mis = s0.detach(), s0_mis.detach()
            extras["energy_goal_own"] = self.energy(g_prev, g_step, g_own, 1)
            extras["energy_goal_mis"] = self.energy(g_prev, g_step, g_mis, 1)
        extras["s0_tgt"] = s0_t
        extras["prev_states_tgt"] = torch.cat(
            [s0_t.unsqueeze(1), step_states_t[:, :-1]], dim=1
        )
        out = JEPAOutputs(
            s0=s0, step_states=step_states, prev_states=prev_states,
            step_states_tgt=step_states_t, actions=actions,
            action_emb_tgt=None, preds=preds, rollout=preds, op_logits=None,
            emb_pred=None, value_pred=None, step_mask=step_mask, extras=extras,
        )
        if self.energy_imagined_rank and "action_cat" in batch:
            self._imagined_energy_rank(out, batch, cat_vecs, s0)
        if "ga_cand_cat" in batch and (batch["ga_t"] >= 0).any():
            self._geo_rank(batch, out, cat_vecs, s_all, s_all_t)
        return out

    # ------------------------------------- energy ranking on imagined states
    def _imagined_energy_rank(self, out, batch, cat_vecs, s0) -> None:
        """Train the Energy head on the DISTRIBUTION DEEP SEARCH QUERIES.

        From every real prefix state s_t, the predictor is rolled h steps
        (h = 1..energy_imagined_depth-1) under the OBSERVED actions
        a_t..a_{t+h-1}, exactly like planning lookahead (``imagine``; no
        re-encoding).  At each imagined state s_hat the observed continuation
        a_{t+h} is contrasted with ``energy_imagined_kcat`` catalogue actions
        sampled uniformly from the problem's own action catalogue (the
        planner's proposal set), excluding the observed one.  Both are pushed
        one further predictor step and scored with
        E(s_t, predictor(s_hat, a), s_0, h+1) -- the identical call the
        planner makes at depth h.  Loss (``EnergyImaginedRank``): the
        continuation that actually occurred must have lower energy.

        Self-supervised: the label is which continuation occurred; no
        symbolic state or feasibility bit is read.  Depth h=0 (the real
        anchor state) is intentionally excluded -- it is covered by
        ``energy_cf_feasibility_rank`` / ``geo_horizon_rank``.
        """
        if self.predictor_kind != "mlp":
            raise NotImplementedError(
                "energy_imagined_rank currently supports predictor_kind=mlp"
            )
        prev_states, actions = out.prev_states, out.actions
        step_mask = out.step_mask
        B, T, D = prev_states.shape
        device = prev_states.device
        kmax = self.energy_imagined_depth
        if kmax < 2:
            return
        Hh = kmax - 1  # imagined depths 1..Hh get a contrast
        pad_a = F.pad(actions, (0, 0, 0, kmax))
        pad_m = F.pad(step_mask.float(), (0, kmax))
        pad_c = F.pad(batch["action_cat"], (0, kmax))
        codes = torch.stack([pad_a[:, j:j + T] for j in range(kmax)], 2)
        steps = torch.stack([pad_m[:, j:j + T] for j in range(kmax)], 2)
        obs_cat = torch.stack([pad_c[:, j:j + T] for j in range(kmax)], 2)
        valid = steps.cumprod(-1) > 0  # [B, T, kmax]
        N = B * T
        _, prefixes = self.imagine(
            prev_states.reshape(N, D),
            codes.reshape(N, kmax, D),
            steps.reshape(N, kmax) > 0,
            return_prefixes=True,
        )  # [N, kmax+1, D]
        pre = prefixes[:, 1:1 + Hh]                      # imagined s_hat at depth h
        exec_codes = codes.reshape(N, kmax, D)[:, 1:]    # observed a_{t+h}
        exec_cat = obs_cat.reshape(N, kmax)[:, 1:]       # its catalogue index
        # depth h contrast needs: a_{t+h} exists AND the whole imagined chain
        # up to it exists (valid[..., h] covers both, cumulative).
        valid_h = valid.reshape(N, kmax)[:, 1:]          # [N, Hh]
        if not bool(valid_h.any()):
            return
        # ---- sample catalogue counterfactuals (uniform over real entries) --
        K = self.energy_imagined_kcat
        cat_mask = batch["cat_mask"]                     # [B, Ncat]
        probs = cat_mask.float().clamp(min=0.0) + 1e-9
        samp = torch.multinomial(
            probs, T * Hh * K, replacement=True
        ).reshape(B, T, Hh, K)
        samp_valid = cat_mask.gather(
            1, samp.reshape(B, -1)
        ).reshape(B, T, Hh, K)
        samp = samp.reshape(N, Hh, K)
        alt_valid = (
            samp_valid.reshape(N, Hh, K)
            & valid_h.unsqueeze(-1)
            & (samp != exec_cat.unsqueeze(-1))
        )
        b_of_n = torch.arange(B, device=device).repeat_interleave(T)
        alt_codes = cat_vecs[b_of_n.view(N, 1, 1), samp]  # [N, Hh, K, D]
        # ---- one predictor step from the imagined state, then the Energy ---
        exec_next = self.predictor(pre, exec_codes)                    # [N, Hh, D]
        alt_next = self.predictor(
            pre.unsqueeze(2).expand(-1, -1, K, -1), alt_codes
        )                                                              # [N, Hh, K, D]
        root_h = prev_states.reshape(N, D).unsqueeze(1).expand(-1, Hh, -1)
        init_h = s0.repeat_interleave(T, 0).unsqueeze(1).expand(-1, Hh, -1)
        horizons = torch.arange(
            2, Hh + 2, device=device, dtype=root_h.dtype
        ).view(1, Hh)  # depth h imagined => endpoint is h+1 steps from root
        e_root, e_init = root_h, init_h
        e_exec_in, e_alt_in = exec_next, alt_next
        if self.value_detach:
            e_root, e_init = root_h.detach(), init_h.detach()
            e_exec_in, e_alt_in = exec_next.detach(), alt_next.detach()
        e_exec = self.energy(e_root, e_exec_in, e_init, horizons)      # [N, Hh]
        e_alt = self.energy(
            e_root.unsqueeze(2).expand(-1, -1, K, -1), e_alt_in,
            e_init.unsqueeze(2).expand(-1, -1, K, -1),
            horizons.unsqueeze(-1),
        )                                                              # [N, Hh, K]
        keep = alt_valid.any(-1).reshape(-1)
        out.extras["energy_img_exec"] = e_exec.reshape(-1)[keep]
        out.extras["energy_img_alt"] = e_alt.reshape(-1, K)[keep]
        out.extras["energy_img_valid"] = alt_valid.reshape(-1, K)[keep]

    # -------------------------------------------------- multi-step rollout
    def _latent_rollout(self, extras, prev_states, actions, step_states_t,
                        step_mask) -> None:
        """k-step latent rollout along the TRUE observed action sequence.

        From every anchor state s_t the predictor is rolled forward k steps
        using the observed actions a_t..a_{t+k-1} (reusing ``imagine``, the
        same recursive machinery planning uses) and the k-step imagined state
        is regressed onto the EMA teacher's true s_{t+k}.  A single rollout of
        length max(ks) yields every prefix, so all k are free.
        """
        B, T, D = prev_states.shape
        ks = self.latent_rollout_ks
        kmax = max(ks)
        pad_a = F.pad(actions, (0, 0, 0, kmax - 1))
        pad_m = F.pad(step_mask.float(), (0, kmax - 1))
        pad_t = F.pad(step_states_t, (0, 0, 0, kmax - 1))
        codes = torch.stack([pad_a[:, j:j + T] for j in range(kmax)], 2)
        steps = torch.stack([pad_m[:, j:j + T] for j in range(kmax)], 2)
        valid = steps.cumprod(-1)  # [B, T, kmax]: all of a_t..a_{t+k-1} exist
        hist_s = hist_a = None
        if self.predictor_kind == "causal":
            # Per-anchor teacher-forced histories: for anchor t the causal
            # predictor grounds on the REAL prefix s_0..s_t / a_0..a_{t-1}.
            # Positions past t are masked to a stalled no-op prefix (anchor
            # state, zero action), mirroring ``_histories``.
            t_idx = torch.arange(T, device=prev_states.device)
            keep_s = t_idx.view(1, T, 1) >= t_idx.view(1, 1, T)   # j <= t
            hist_s = torch.where(
                keep_s.unsqueeze(-1),
                prev_states.unsqueeze(1).expand(B, T, T, D),
                prev_states.unsqueeze(2).expand(B, T, T, D),
            ).reshape(B * T, T, D)
            keep_a = t_idx.view(1, T, 1) > t_idx.view(1, 1, T)    # j < t
            hist_a = torch.where(
                keep_a.unsqueeze(-1),
                actions.unsqueeze(1).expand(B, T, T, D),
                torch.zeros(1, 1, 1, D, dtype=actions.dtype,
                            device=actions.device),
            ).reshape(B * T, T, D)
            # keep only a_0..a_{t-1}: drop the last slot, which is never a
            # valid past action for any anchor (a_{T-1} belongs to the future
            # of every anchor except t=T-1, whose own slot is zeroed anyway).
            hist_a = hist_a[:, : T - 1]
            hist_s = hist_s[:, : T]
        _, prefixes = self.imagine(
            prev_states.reshape(B * T, D),
            codes.reshape(B * T, kmax, D),
            steps.reshape(B * T, kmax) > 0,
            hist_s, hist_a,
            return_prefixes=True,
        )
        prefixes = prefixes.reshape(B, T, kmax + 1, D)
        preds, tgts, masks = [], [], []
        for k in ks:
            preds.append(prefixes[:, :, k])
            tgts.append(pad_t[:, k - 1:k - 1 + T])
            masks.append(valid[:, :, k - 1])
        extras["rollout_preds"] = torch.stack(preds, 2)     # [B, T, K, D]
        extras["rollout_targets"] = torch.stack(tgts, 2).detach()
        extras["rollout_valid"] = torch.stack(masks, 2)     # [B, T, K]
        extras["rollout_ks"] = ks
        with torch.no_grad():
            for i, k in enumerate(ks):
                cos = F.cosine_similarity(
                    _ln(preds[i].float()), _ln(tgts[i].float()), dim=-1
                )
                m = masks[i]
                extras[f"diag_rollout_cos_k{k}"] = (
                    (cos * m).sum() / m.sum().clamp(min=1.0)
                )

    # ----------------------------------------------------------- geo rank
    def _geo_rank(self, batch, out, cat_vecs, s_all, s_all_t) -> None:
        tokens = batch["tokens"]
        B = tokens.shape[0]
        device = tokens.device
        bidx = torch.arange(B, device=device)
        valid_b = batch["ga_t"] >= 0
        t = batch["ga_t"].clamp(min=0)
        s_anchor = s_all[bidx, t]  # online state before the anchor step
        s0 = out.s0
        cand_cat = batch["ga_cand_cat"]  # [B, C]
        cand_valid = batch["ga_cand_valid"] & valid_b.unsqueeze(1)
        C = cand_cat.shape[1]
        K = C - 1
        a_cand = cat_vecs[bidx.unsqueeze(1), cand_cat]  # [B, C, D]
        hist_s = hist_a = None
        if self.predictor_kind == "causal":
            hist_s, hist_a = self._histories(out, t)
            flat_hist_s = hist_s.repeat_interleave(C, 0)
            flat_hist_a = hist_a.repeat_interleave(C, 0)
            preds_cand = self.predict(
                s_anchor.repeat_interleave(C, 0), a_cand.reshape(B * C, -1),
                flat_hist_s, flat_hist_a,
            ).reshape(B, C, -1)
        else:
            preds_cand = self.predictor(
                s_anchor.unsqueeze(1).expand(-1, C, -1), a_cand
            )
        pe, preds_alt = preds_cand[:, 0], preds_cand[:, 1:]
        root_c = s_anchor.unsqueeze(1).expand(-1, C, -1)
        if self.value_detach:
            energies = self.energy(
                root_c.detach(), preds_cand.detach(), s0.detach(), 1
            )
        else:
            energies = self.energy(root_c, preds_cand, s0, 1)  # [B, C]
        if self.energy_cf_feasibility_rank and K > 0:
            cf_valid = cand_valid[:, 1:]
            if self.energy_cf_scope == "invalid_only":
                cf_valid = cf_valid & batch["ga_alt_invalid"]
            out.extras["energy_cf_exec"] = energies[:, 0]
            out.extras["energy_cf_alt"] = energies[:, 1:]
            out.extras["energy_cf_valid"] = cf_valid
        # ---- EMA teacher labels from TRUE rollout text -------------------
        with torch.no_grad():
            last = out.step_mask.sum(1).clamp(min=1)
            goal = s_all_t[bidx, last]  # encoded solved state
            rt = batch["ga_roll_tokens"]  # [B, C, R, Lr]
            _, _, R, Lr = rt.shape
            rv = batch["ga_roll_valid"] & valid_b.view(B, 1, 1)
            flat = rt.reshape(B * C * R, Lr)
            active = rv.reshape(-1)
            positions = torch.stack(
                [batch["ga_roll_first"].reshape(-1),
                 batch["ga_roll_leaf"].reshape(-1)], dim=1
            )
            gathered = torch.zeros(
                B * C * R, 2, self.d_model, device=device, dtype=s0.dtype
            )
            if active.any():
                gathered[active] = self.teacher_gather(
                    flat[active], positions[active]
                ).to(gathered.dtype)
            first = gathered[:, 0].reshape(B, C, R, -1)
            leaf = gathered[:, 1].reshape(B, C, R, -1)
            d_rollout = (_ln(leaf) - _ln(goal).view(B, 1, 1, -1)).abs().mean(-1)
            d_rollout = d_rollout.masked_fill(~rv, float("inf"))
            d = d_rollout.amin(-1)  # best-of-R per candidate
            label_valid = rv.any(-1) & cand_valid
            # true next states of the alternatives (first rollout of each)
            s_alt_true = first[:, 1:, 0]  # [B, K, D]
        # ---- one-step ranking + pair-difference regression ---------------
        out.extras["ga_energy"] = energies
        out.extras["ga_label"] = d
        out.extras["ga_valid"] = label_valid
        out.extras["ga_cf_pred"] = preds_alt
        out.extras["ga_cf_target"] = s_alt_true
        out.extras["ga_cf_valid"] = cand_valid[:, 1:] & rv[:, 1:, 0]
        # ---- horizon endpoint ranking over imagined rollouts --------------
        act = batch["ga_roll_act"]  # [B, C, R, H]
        act_mask = batch["ga_roll_act_mask"] & rv.unsqueeze(-1)
        Hh = act.shape[-1]
        codes = cat_vecs[
            bidx.view(B, 1, 1, 1), act
        ].reshape(B * C * R, Hh, -1)
        root = s_anchor.unsqueeze(1).unsqueeze(1).expand(-1, C, R, -1).reshape(
            B * C * R, -1
        )
        flat_mask = act_mask.reshape(B * C * R, Hh)
        if self.predictor_kind == "causal":
            endpoint, prefixes = self.imagine(
                root, codes, flat_mask,
                hist_s.repeat_interleave(C * R, 0),
                hist_a.repeat_interleave(C * R, 0), return_prefixes=True,
            )
        else:
            endpoint, prefixes = self.imagine(
                root, codes, flat_mask, return_prefixes=True
            )
        initial = s0.unsqueeze(1).unsqueeze(1).expand(-1, C, R, -1).reshape(
            B * C * R, -1
        )
        horizon = batch["ga_requested_horizon"].view(B, 1, 1).expand(
            B, C, R
        ).reshape(-1)
        e_root, e_end, e_init = root, endpoint, initial
        if self.value_detach:
            e_root, e_end, e_init = root.detach(), endpoint.detach(), initial.detach()
        out.extras["ga_horizon_energy"] = self.energy(
            e_root, e_end, e_init, horizon
        ).reshape(B, C, R)
        out.extras["ga_horizon_label"] = d_rollout
        out.extras["ga_horizon_valid"] = rv & act_mask.any(-1)
        # ---- legality contrast ALONG imagined prefixes ---------------------
        # At rollout depth h >= 1 the observed continuation is rollout action
        # h (a feasible action that actually occurred in the teacher rollout)
        # and the counterfactuals are infeasible intents at that rollout
        # state (ga_roll_cf).  Both are imagined from the SAME imagined
        # prefix and scored by the SAME endpoint Energy with the real anchor
        # as root -- exactly how oracle-free lookahead scores deeper slots.
        if (
            self.energy_cf_feasibility_rank
            and "ga_roll_cf" in batch
            and batch["ga_roll_cf_mask"].any()
        ):
            cf = batch["ga_roll_cf"]  # [B, C, R, H, Kc]
            cf_mask = batch["ga_roll_cf_mask"] & act_mask.unsqueeze(-1)
            Kc = cf.shape[-1]
            depth_ok = torch.zeros(Hh, dtype=torch.bool, device=device)
            depth_ok[1:] = True  # depth 0 is the anchor contrast above
            cf_mask = cf_mask & depth_ok.view(1, 1, 1, Hh, 1)
            if cf_mask.any():
                N = B * C * R
                cf_codes = cat_vecs[
                    bidx.view(B, 1, 1, 1, 1), cf
                ].reshape(N, Hh, Kc, -1)
                pre = prefixes[:, :Hh]  # [N, H, D] state before action h
                # one-step imagination from every imagined prefix (Markov
                # call; for the causal predictor option this is the Markov
                # approximation of the depth contrast).
                if self.predictor_kind == "causal":
                    Dm = pre.shape[-1]
                    exec_next = self.predictor(
                        pre.reshape(-1, Dm), codes[:, :Hh].reshape(-1, Dm)
                    ).reshape(N, Hh, Dm)
                    cf_next = self.predictor(
                        pre.unsqueeze(2).expand(-1, -1, Kc, -1)
                        .reshape(-1, Dm),
                        cf_codes.reshape(-1, Dm),
                    ).reshape(N, Hh, Kc, Dm)
                else:
                    exec_next = self.predictor(pre, codes)  # [N, H, D]
                    cf_next = self.predictor(
                        pre.unsqueeze(2).expand(-1, -1, Kc, -1), cf_codes
                    )  # [N, H, Kc, D]
                root_h = root.unsqueeze(1).expand(-1, Hh, -1)
                init_h = initial.unsqueeze(1).expand(-1, Hh, -1)
                e_exec_d = self.energy(root_h, exec_next, init_h, 1)  # [N, H]
                e_cf_d = self.energy(
                    root_h.unsqueeze(2).expand(-1, -1, Kc, -1), cf_next,
                    init_h.unsqueeze(2).expand(-1, -1, Kc, -1), 1,
                )  # [N, H, Kc]
                valid_d = cf_mask.reshape(N, Hh, Kc)
                keep = valid_d.any(-1).reshape(-1)  # [N*H]
                exec_flat = e_exec_d.reshape(-1)[keep]
                cf_flat = e_cf_d.reshape(-1, Kc)[keep]
                valid_flat = valid_d.reshape(-1, Kc)[keep]
                prev_exec = out.extras.get("energy_cf_exec")
                if prev_exec is not None:
                    prev_alt = out.extras["energy_cf_alt"]
                    prev_valid = out.extras["energy_cf_valid"]
                    Kmax = max(prev_alt.shape[1], Kc)
                    prev_alt = F.pad(prev_alt, (0, Kmax - prev_alt.shape[1]))
                    prev_valid = F.pad(prev_valid, (0, Kmax - prev_valid.shape[1]))
                    cf_flat = F.pad(cf_flat, (0, Kmax - Kc))
                    valid_flat = F.pad(valid_flat, (0, Kmax - Kc))
                    out.extras["energy_cf_exec"] = torch.cat([prev_exec, exec_flat])
                    out.extras["energy_cf_alt"] = torch.cat([prev_alt, cf_flat])
                    out.extras["energy_cf_valid"] = torch.cat([prev_valid, valid_flat])
                else:
                    out.extras["energy_cf_exec"] = exec_flat
                    out.extras["energy_cf_alt"] = cf_flat
                    out.extras["energy_cf_valid"] = valid_flat
                out.extras["energy_cf_depth_pairs"] = valid_flat.sum()
                with torch.no_grad():
                    # diagnostic: fraction of depth pairs already ordered
                    ordered = (exec_flat.unsqueeze(1) < cf_flat) & valid_flat
                    out.extras["energy_cf_depth_acc"] = (
                        ordered.sum().float() / valid_flat.sum().clamp(min=1)
                    )

        # ---- PARTIAL-TRAJECTORY (prefix) ranking ---------------------------
        if self.energy_prefix_rank and "ga_roll_cf" in batch:
            if self.predictor_kind == "causal":
                hist_s_n = hist_s.repeat_interleave(C * R, 0)
                hist_a_n = hist_a.repeat_interleave(C * R, 0)
            else:
                hist_s_n = hist_a_n = None
            self._prefix_rank(
                out, batch, cat_vecs, root, initial, codes, flat_mask,
                prefixes, act_mask, rv, B, C, R, Hh, hist_s_n, hist_a_n,
            )

        # ---- hindsight long-horizon endpoint ranking -----------------------
        if self.hindsight_long_horizon_rank and "ga_hl_act" in batch:
            self._hindsight_long_rank(
                out, batch, cat_vecs, s_anchor, goal, t, valid_b,
                hist_s, hist_a,
            )

    # ---------------------------------------- hindsight long-horizon rank
    def _hindsight_long_rank(self, out, batch, cat_vecs, s_anchor, goal, t,
                             valid_b, hist_s=None, hist_a=None) -> None:
        """Long-horizon endpoint ranking against the HINDSIGHT goal.

        From the anchor state s_t the predictor imagines (a) the OBSERVED
        continuation a_t..a_{t+h-1} and (b) random feasible same-length
        rollouts from the dataset (``ga_hl_act``), with h drawn uniformly up
        to the FULL remaining trajectory length -- not the geo_rank_horizons
        cap, which concentrates every other contrast near the terminal.  The
        Energy's goal/initial slot is the trajectory's own achieved terminal
        (teacher-encoded), i.e. hindsight relabeling: the trajectory
        demonstrably reached it, so no symbolic label is read.  Loss
        (``HindsightLongHorizonRank``): the observed continuation's endpoint
        must get lower energy than the random walks' endpoints.
        """
        actions, step_mask = out.actions, out.step_mask
        B, T, D = actions.shape
        device = actions.device
        bidx = torch.arange(B, device=device)
        h = batch["ga_hl_h"].clamp(min=1)                      # [B]
        hl_valid = (batch["ga_hl_h"] > 0) & valid_b            # [B]
        Hl = max(int(h.max().item()), 1)
        ar = torch.arange(Hl, device=device).view(1, Hl)
        idx = (t.unsqueeze(1) + ar).clamp(max=T - 1)
        obs_codes = actions[bidx.unsqueeze(1), idx]            # [B, Hl, D]
        obs_mask = (
            (ar < h.unsqueeze(1)) & step_mask.gather(1, idx)
            & hl_valid.unsqueeze(1)
        )
        pos_end = self.imagine(s_anchor, obs_codes, obs_mask, hist_s, hist_a)
        neg = batch["ga_hl_act"]                               # [B, Rl, Hn]
        neg_mask = batch["ga_hl_act_mask"] & hl_valid.view(B, 1, 1)
        Rl, Hn = neg.shape[1], neg.shape[2]
        neg_codes = cat_vecs[bidx.view(B, 1, 1), neg]          # [B, Rl, Hn, D]
        neg_end = self.imagine(
            s_anchor.repeat_interleave(Rl, 0),
            neg_codes.reshape(B * Rl, Hn, D),
            neg_mask.reshape(B * Rl, Hn),
            hist_s.repeat_interleave(Rl, 0) if hist_s is not None else None,
            hist_a.repeat_interleave(Rl, 0) if hist_a is not None else None,
        ).reshape(B, Rl, D)
        hz = h.to(s_anchor.dtype)
        e_root, e_pos, e_neg = s_anchor, pos_end, neg_end
        if self.value_detach:
            e_root, e_pos, e_neg = (
                s_anchor.detach(), pos_end.detach(), neg_end.detach()
            )
        out.extras["energy_hl_obs"] = self.energy(e_root, e_pos, goal, hz)
        out.extras["energy_hl_neg"] = self.energy(
            e_root.unsqueeze(1).expand(-1, Rl, -1), e_neg, goal,
            hz.unsqueeze(1),
        )
        out.extras["energy_hl_valid"] = (
            neg_mask.any(-1) & obs_mask.any(-1).unsqueeze(1)
        )

    # ------------------------------------------------- prefix (path) rank
    def _prefix_rank(self, out, batch, cat_vecs, root, initial, codes,
                     flat_mask, prefixes, act_mask, rv, B, C, R, Hh,
                     hist_s_n=None, hist_a_n=None) -> None:
        """Rank whole imagined PATHS, not just their endpoints.

        The observed continuation of a rollout is the action sequence that
        actually occurred (a_1..a_H).  A counterfactual PATH is the same
        sequence with one counterfactual intent INSERTED at a random depth
        j >= 1 (``ga_roll_cf``, the infeasible intents the dataset already
        samples at every rollout state) and the last true action dropped, so
        both paths are the same length.  Because an infeasible intent is a
        no-op in the environment, the inserted path simply WASTES one of its
        imagined steps -- exactly the degeneracy an endpoint-only Energy
        cannot see.  The head therefore emits an energy for EVERY
        prefix of both paths, E(root, prefix_j, s_0, j), and the objective
        aggregates them into one path score before the pairwise contrast.

        The only label is "which continuation actually occurred", which is in
        the data.  No symbolic state, step count, or ranking label is read.
        """
        device = root.device
        D = root.shape[-1]
        N = root.shape[0]
        cf = batch["ga_roll_cf"].reshape(N, Hh, -1)
        cf_mask = (
            batch["ga_roll_cf_mask"] & act_mask.unsqueeze(-1)
        ).reshape(N, Hh, -1)
        Kc = cf.shape[-1]
        # depth 0 is the anchor decision (already contrasted by
        # energy_cf_feasibility_rank); insertions live strictly inside the
        # imagined continuation.
        depth_ok = torch.zeros(Hh, dtype=torch.bool, device=device)
        depth_ok[1:] = True
        cf_mask = cf_mask & depth_ok.view(1, Hh, 1)
        # HARDER NEGATIVES (opt-in).  The dataset tags each rollout
        # counterfactual as 1 = premature (its parents are not resolved yet)
        # or 2 = already resolved (re-deriving a fact the state already
        # contains -- it still reads as a sensible sentence, so telling it
        # apart from a step that made progress is the harder judgement).
        # This selects WHICH negatives are offered; the ranking label stays
        # "which continuation actually occurred".
        if self.energy_prefix_cf_kind != "all":
            kinds = batch.get("ga_roll_cf_kind")
            if kinds is None:
                raise KeyError(
                    "energy_prefix_cf_kind needs ga_roll_cf_kind from the "
                    "dataset; rebuild the stream with a current snapshot"
                )
            want = 1 if self.energy_prefix_cf_kind == "premature" else 2
            cf_mask = cf_mask & (kinds.reshape(N, Hh, -1) == want)
        if not bool(cf_mask.any()):
            return

        # ---- energies of every prefix of the OBSERVED path ---------------
        depths = torch.arange(1, Hh + 1, device=device, dtype=root.dtype)
        root_h = root.unsqueeze(1).expand(-1, Hh, -1)
        init_h = initial.unsqueeze(1).expand(-1, Hh, -1)
        e_obs = self.energy(root_h, prefixes[:, 1:], init_h, depths.view(1, Hh))

        # ---- one random insertion depth per counterfactual slot -----------
        valid_nk = cf_mask.permute(0, 2, 1).contiguous()      # [N, Kc, Hh]
        has_pair = valid_nk.any(-1)                            # [N, Kc]
        weights = valid_nk.float()
        if self.energy_prefix_depth_bias == "late":
            # Bias the wasted step toward the END of the imagined path, where
            # its consequence is subtler and less of the remaining trajectory
            # is left to expose it.  Weight grows linearly with depth.
            weights = weights * torch.arange(
                1, Hh + 1, device=device, dtype=weights.dtype
            ).view(1, 1, Hh)
        probs = torch.where(
            has_pair.unsqueeze(-1), weights,
            torch.ones_like(weights),
        )
        j = torch.multinomial(probs.reshape(N * Kc, Hh), 1).reshape(N, Kc)
        cf_pick = cf.permute(0, 2, 1).gather(2, j.unsqueeze(-1)).squeeze(-1)
        b_of_n = torch.arange(B, device=device).repeat_interleave(C * R)
        cf_vec = cat_vecs[b_of_n.unsqueeze(1), cf_pick]        # [N, Kc, D]

        # ---- build the inserted action sequence (LENGTH-MATCHED) ----------
        # [a_0 .. a_{j-1}, cf, a_j .. a_{Hh-2}]: the junk step displaces the
        # last true action instead of being appended, so the counterfactual
        # path spends exactly as many imagined steps as the observed one.
        # This matters: at planning time every candidate is rolled out to the
        # same depth, so a score that separated the paths by LENGTH would
        # transfer nothing.
        L = Hh
        ar = torch.arange(L, device=device).view(1, 1, L)
        src = torch.where(ar < j.unsqueeze(-1), ar, ar - 1).clamp(min=0)
        seq = codes.gather(
            1, src.reshape(N, Kc * L, 1).expand(N, Kc * L, D)
        ).reshape(N, Kc, L, D)
        msk = flat_mask.gather(1, src.reshape(N, Kc * L)).reshape(N, Kc, L)
        seq = seq.scatter(
            2, j.view(N, Kc, 1, 1).expand(N, Kc, 1, D), cf_vec.unsqueeze(2)
        )
        msk = msk.scatter(
            2, j.unsqueeze(-1), torch.ones_like(j.unsqueeze(-1), dtype=msk.dtype)
        )
        # ---- OPTIONAL EXTRA INSERTIONS (harder negatives) -----------------
        # ``energy_prefix_n_insert`` > 1 wastes several steps per
        # counterfactual path instead of one: each extra insertion shifts the
        # suffix right (dropping one more true action at the tail) and
        # scatters a fresh infeasible intent at a freshly sampled depth.
        # Deep beam candidates differ from the observed path in many steps;
        # single-insert negatives never cover that regime.
        for _extra in range(1, self.energy_prefix_n_insert):
            j_i = torch.multinomial(
                probs.reshape(N * Kc, Hh), 1
            ).reshape(N, Kc).clamp(max=L - 1)
            pick_i = cf.permute(0, 2, 1).gather(
                2, j_i.unsqueeze(-1)).squeeze(-1)
            cf_vec_i = cat_vecs[b_of_n.unsqueeze(1), pick_i]
            pos = ar.expand(N, Kc, L)
            shift_src = (pos - 1).clamp(min=0)
            seq_shift = seq.gather(
                2, shift_src.unsqueeze(-1).expand(N, Kc, L, D))
            msk_shift = msk.gather(2, shift_src)
            take_new = pos >= j_i.unsqueeze(-1)
            seq = torch.where(take_new.unsqueeze(-1), seq_shift, seq)
            msk = torch.where(take_new, msk_shift, msk)
            seq = seq.scatter(
                2, j_i.view(N, Kc, 1, 1).expand(N, Kc, 1, D),
                cf_vec_i.unsqueeze(2))
            msk = msk.scatter(
                2, j_i.unsqueeze(-1),
                torch.ones_like(j_i.unsqueeze(-1), dtype=msk.dtype))
        # padded rows have spare slots after the true actions; truncate so the
        # counterfactual path never buys back the step it wasted.
        n_obs = flat_mask.sum(-1, keepdim=True).unsqueeze(-1)   # [N, 1, 1]
        msk = msk & (ar < n_obs)

        # ---- imagine the counterfactual paths and score every prefix ------
        M = N * Kc
        root_k = root.unsqueeze(1).expand(N, Kc, D).reshape(M, D)
        _, cf_pre = self.imagine(
            root_k, seq.reshape(M, L, D), msk.reshape(M, L),
            hist_s_n.repeat_interleave(Kc, 0) if hist_s_n is not None else None,
            hist_a_n.repeat_interleave(Kc, 0) if hist_a_n is not None else None,
            return_prefixes=True,
        )
        depths_l = torch.arange(1, L + 1, device=device, dtype=root.dtype)
        root_l = root_k.unsqueeze(1).expand(M, L, D)
        init_l = initial.unsqueeze(1).expand(N, Kc, D).reshape(M, 1, D).expand(
            M, L, D
        )
        e_cf = self.energy(
            root_l, cf_pre[:, 1:], init_l, depths_l.view(1, L)
        ).reshape(N, Kc, L)

        out.extras["energy_prefix_obs"] = e_obs                # [N, Hh]
        out.extras["energy_prefix_obs_valid"] = flat_mask      # [N, Hh]
        out.extras["energy_prefix_cf"] = e_cf                  # [N, Kc, L]
        out.extras["energy_prefix_cf_valid"] = msk             # [N, Kc, L]
        with torch.no_grad():
            # where in the path the wasted step was inserted (1-indexed);
            # confirms energy_prefix_depth_bias is doing what it claims
            pv0 = has_pair & rv.reshape(N, 1)
            out.extras["diag_energy_prefix_ins_depth"] = (
                (j.float() * pv0).sum() / pv0.sum().clamp(min=1)
            )
        out.extras["energy_prefix_pair_valid"] = (
            has_pair & rv.reshape(N, 1) & flat_mask.any(-1, keepdim=True)
        )

    def _histories(self, out, t):
        """Teacher-forced prefix (s_0..s_t, a_0..a_{t-1}) padded per row for
        the causal predictor; positions after t are masked by truncation to
        the max anchor (the predictor is causal, so extra suffix is harmless
        only if we cut at t: we therefore zero actions beyond t)."""
        states_all = torch.cat([out.s0.unsqueeze(1), out.step_states], 1)
        tmax = int(t.max().item())
        hist_s = states_all[:, : tmax + 1]
        hist_a = out.actions[:, :tmax]
        # rows with smaller t: the extra history would leak the future.  We
        # therefore mask by replacing with the anchor state/zero action;
        # a causal predictor reading them sees a stalled no-op prefix.
        B = t.shape[0]
        ar = torch.arange(tmax + 1, device=t.device).view(1, -1)
        keep_s = ar <= t.view(B, 1)
        anchor = states_all[torch.arange(B, device=t.device), t]
        hist_s = torch.where(keep_s.unsqueeze(-1), hist_s, anchor.unsqueeze(1))
        keep_a = ar[:, :tmax] < t.view(B, 1)
        hist_a = torch.where(keep_a.unsqueeze(-1), hist_a, torch.zeros_like(hist_a))
        return hist_s, hist_a

    # ------------------------------------------------------------ planning
    @torch.no_grad()
    def encode_history(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[1, L] token history -> (hidden [1, L, D], state at last token)."""
        h = self.encode(history)
        return h, h[:, -1]

    @torch.no_grad()
    def encode_candidates_in_context(
        self, history: list[int], phrases: list[list[int]], device
    ) -> torch.Tensor:
        """Intent phrases encoded after ``history`` with the phrase block:
        returns [n, D] action vectors (hidden at each phrase's last token)."""
        n = len(phrases)
        P = len(history)
        total = P + sum(len(p) for p in phrases)
        tokens = torch.full((1, total), self.pad_id, dtype=torch.long)
        pos = torch.zeros((1, total), dtype=torch.long)
        pid = torch.full((1, total), -1, dtype=torch.long)
        tokens[0, :P] = torch.tensor(history)
        pos[0, :P] = torch.arange(P)
        last = []
        cur = P
        for j, p in enumerate(phrases):
            k = len(p)
            tokens[0, cur:cur + k] = torch.tensor(p)
            pos[0, cur:cur + k] = P + torch.arange(k)
            pid[0, cur:cur + k] = j
            last.append(cur + k - 1)
            cur += k
        tokens, pos, pid = tokens.to(device), pos.to(device), pid.to(device)
        allowed = build_block_attention(
            tokens, self.pad_id, torch.tensor([P], device=device),
            torch.tensor([P - 1], device=device), pid,
        )
        h = self.encode(tokens, allowed, pos)
        return h[0, torch.tensor(last, device=device)]
