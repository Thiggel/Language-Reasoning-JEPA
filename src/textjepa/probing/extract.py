"""Extract frozen latents + ground-truth labels for offline probing."""

from __future__ import annotations

import numpy as np
import torch

from textjepa.training.trainer import to_device


@torch.no_grad()
def extract_features(
    model, loader, device: torch.device, max_batches: int | None = None
) -> dict[str, np.ndarray]:
    model.eval()
    step_keys = ("state", "pred", "rollout", "delta", "action")
    step_labels = ("op", "value", "remaining", "resolved_n", "necessary")
    sample_keys = ("s0", "final_state")
    sample_labels = ("answer", "n_necessary", "n_vars")
    acc: dict[str, list[np.ndarray]] = {
        k: [] for k in step_keys + step_labels + sample_keys + sample_labels
    }

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = to_device(batch, device)
        out = model(batch)
        m = out.step_mask.reshape(-1)
        flat = lambda x: x.reshape(-1, x.shape[-1])[m].cpu().numpy()
        acc["state"].append(flat(out.step_states))
        acc["pred"].append(flat(out.preds))
        acc["rollout"].append(flat(out.rollout))
        acc["delta"].append(flat(out.step_states - out.prev_states))
        acc["action"].append(flat(out.actions))
        if "step_tokens" in batch:
            acc.setdefault("chunk_emb", []).append(
                flat(model.encode_chunks(batch["step_tokens"]))
            )
        if "chunk_pred" in out.extras:
            acc.setdefault("chunk_pred", []).append(flat(out.extras["chunk_pred"]))
            acc.setdefault("chunk_pred_rollout", []).append(
                flat(out.extras["chunk_pred_rollout"])
            )
        for k in step_labels:
            acc[k].append(batch[k].reshape(-1)[m].cpu().numpy())
        v = batch["value"]
        for lag in (1, 2, 3):
            pv = torch.cat([torch.full_like(v[:, :lag], -1), v[:, :-lag]], dim=1)
            acc.setdefault(f"value_prev{lag}", []).append(
                pv.reshape(-1)[m].cpu().numpy()
            )
        last = out.step_mask.sum(dim=1) - 1
        b_idx = torch.arange(out.step_states.shape[0], device=device)
        acc["s0"].append(out.s0.cpu().numpy())
        acc["final_state"].append(out.step_states[b_idx, last].cpu().numpy())
        for k in sample_labels:
            acc[k].append(batch[k].cpu().numpy())
        # answer replicated to every step (emergence-over-time probe)
        ans = batch["answer"].unsqueeze(1).expand_as(out.step_mask)
        acc.setdefault("answer_step", []).append(
            ans.reshape(-1)[m].cpu().numpy()
        )
        if "var_idx" in batch:
            _extract_igsm_structure(acc, batch, out, m)
        if "defect_mask" in batch:
            _extract_edit_structure(acc, batch, out)

    feats = {k: np.concatenate(v) for k, v in acc.items() if v}
    # circular value coding targets (mod-p values live on a circle?)
    if "value" in feats:
        modulus = int(feats["value"].max()) + 1
        ang = 2 * np.pi * feats["value"] / modulus
        feats["value_cos"] = np.cos(ang).astype(np.float32)
        feats["value_sin"] = np.sin(ang).astype(np.float32)
    return feats


MAX_VARS = 12
MAX_POS = 16


def _onehot(idx: torch.Tensor, width: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(idx.clamp(min=0), width).float()


def _extract_igsm_structure(acc, batch, out, m) -> None:
    """Binding probes: [state; onehot(var)] rows for membership questions."""
    B, T = out.step_mask.shape
    device = out.step_states.device
    acc.setdefault("var_idx", []).append(
        batch["var_idx"].reshape(-1)[m].cpu().numpy()
    )
    acc.setdefault("query_idx", []).append(batch["query_idx"].cpu().numpy())
    # resolved-set membership: for each valid state sample 4 variables
    g = torch.Generator(device="cpu").manual_seed(int(batch["index"][0]))
    n_vars = batch["n_vars"]  # [B]
    var_idx = batch["var_idx"]  # [B, T]
    rows, labels = [], []
    for k in range(4):
        j = (torch.rand(B, T, generator=g) * n_vars.cpu().unsqueeze(1)).long()
        j = j.clamp(max=MAX_VARS - 1).to(device)
        # resolved at state t  <=>  j appears in var_idx[:, : t + 1]
        seen = (var_idx.unsqueeze(1) == j.unsqueeze(2)) & (
            torch.arange(T, device=device).unsqueeze(0).unsqueeze(0)
            <= torch.arange(T, device=device).view(1, T, 1)
        )
        member = seen.any(dim=2).long()  # [B, T]
        feat = torch.cat(
            [out.step_states, _onehot(j, MAX_VARS)], dim=-1
        )
        rows.append(feat.reshape(-1, feat.shape[-1])[m].cpu().numpy())
        labels.append(member.reshape(-1)[m].cpu().numpy())
    acc.setdefault("state_var", []).append(np.concatenate(rows))
    acc.setdefault("resolved_member", []).append(np.concatenate(labels))
    # relevance map: [s0; onehot(var)] -> is var an ancestor of the query
    anc = batch["ancestor_mask"]  # [B, MAX_VARS]
    s0_rows, s0_labels = [], []
    for j in range(MAX_VARS):
        valid = j < n_vars
        if valid.sum() == 0:
            continue
        oh = torch.zeros(int(valid.sum()), MAX_VARS, device=device)
        oh[:, j] = 1.0
        s0_rows.append(
            torch.cat([out.s0[valid], oh], dim=-1).cpu().numpy()
        )
        s0_labels.append(anc[valid, j].cpu().numpy())
    acc.setdefault("s0_var", []).append(np.concatenate(s0_rows))
    acc.setdefault("ancestor_member", []).append(np.concatenate(s0_labels))


def _extract_edit_structure(acc, batch, out) -> None:
    """[state; onehot(position)] rows for per-position defect probes."""
    dm = batch["defect_mask"]  # [B, T, MAX_POS], -1 = absent
    B, T, P = dm.shape
    device = out.step_states.device
    acc.setdefault("edit_pos", []).append(
        batch["edit_pos"].reshape(-1)[out.step_mask.reshape(-1)].cpu().numpy()
    )
    states = out.step_states.unsqueeze(2).expand(B, T, P, -1)
    oh = torch.eye(MAX_POS, device=device).view(1, 1, P, P).expand(B, T, P, P)
    valid = (dm >= 0) & out.step_mask.unsqueeze(-1)
    feat = torch.cat([states, oh], dim=-1)[valid]
    acc.setdefault("state_pos", []).append(feat.cpu().numpy())
    acc.setdefault("defect_member", []).append(dm[valid].cpu().numpy())
