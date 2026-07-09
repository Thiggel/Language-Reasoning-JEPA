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
        for lag in (1, 2):
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

    return {k: np.concatenate(v) for k, v in acc.items()}
