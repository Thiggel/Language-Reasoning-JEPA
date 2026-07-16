"""Audit whether latent or observed actions carry transition semantics.

This is deliberately separate from planning.  It probes posterior and prior
action means on the solution-sentence transitions only, compares prediction
with matched versus shuffled action codes, and measures posterior/prior
retrieval.  Usage::

    .venv/bin/python scripts/probe_variational.py \
        --ckpt runs/sentence_vjepa_faithful/best.pt --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from textjepa.probing.probes import ridge_probe_accuracy
from textjepa.training.trainer import to_device
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset, collate_for, load_run
from textjepa.utils.metrics import effective_rank, feature_std


def normalized_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.layer_norm(x, x.shape[-1:])
    y = F.layer_norm(y, y.shape[-1:])
    return (x - y).abs().mean(-1)


def solution_transition_indices(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return batch and transition indices for real solution steps.

    In the packed sentence stream the first solution transition connects the
    last prompt sentence to the first solution sentence, hence ``P - 1``.
    """
    bs, ts = [], []
    for b in range(batch["prompt_mask"].shape[0]):
        p = int(batch["prompt_mask"][b].sum().item())
        n = int(batch["step_mask"][b].sum().item())
        bs.extend([b] * n)
        ts.extend(range(p - 1, p - 1 + n))
    device = batch["prompt_mask"].device
    return (
        torch.tensor(bs, dtype=torch.long, device=device),
        torch.tensor(ts, dtype=torch.long, device=device),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prior-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()
    seed_everything(args.seed)

    model, vocab, cfg = load_run(args.ckpt, args.device)
    if model.__class__.__name__ not in {"SentenceStreamVJEPA", "DiscourseVJEPA"}:
        raise TypeError("probe_variational requires a variational JEPA checkpoint")
    dataset = build_dataset(
        cfg, vocab, split="val", size=min(args.n_samples, cfg.data.val_size)
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=partial(collate_for(cfg), pad_id=vocab.pad_id),
    )

    action_mode = getattr(model, "action_mode", "latent")
    if action_mode == "latent":
        features: dict[str, list[torch.Tensor]] = {
            "posterior": [], "prior": [], "sample": [], "delta": [],
        }
    else:
        # In observed-action DiscourseVJEPA, q_mu == p_mu == actions by
        # construction. Calling these posterior/prior features would imply a
        # latent inference problem that the model does not solve.
        features = {"observed_action": [], "delta": []}
    labels: dict[str, list[torch.Tensor]] = {
        "op": [], "necessary": [], "value": [],
    }
    own_error, shuffled_error = [], []
    prior_error, shuffled_prior_error = [], []
    prior_best_error, shuffled_prior_best_error = [], []
    prior_posterior_cosine = []
    prior_component_entropy, prior_component_max = [], []
    prior_component_probabilities = []
    predictive_nll, z_squared, cover1, cover2 = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, torch.device(args.device))
            out = model(batch)
            if out.extras.get("discourse_variational", False):
                bidx, tidx = batch["step_mask"].nonzero(as_tuple=True)
            else:
                bidx, tidx = solution_transition_indices(batch)
            q = out.extras["action_q_mu"][bidx, tidx]
            p = out.extras["action_p_mu"][bidx, tidx]
            a = out.actions[bidx, tidx]
            prev = out.prev_states[bidx, tidx]
            target = out.step_states_tgt[bidx, tidx]
            delta = out.step_states[bidx, tidx] - prev
            pred, _ = model.transition(prev, q)
            permutation = torch.roll(torch.arange(q.shape[0], device=q.device), 1)
            pred_shuffled, _ = model.transition(prev, q[permutation])
            if action_mode == "latent":
                pred_prior, _ = model.transition(prev, p)
                pred_prior_shuffled, _ = model.transition(prev, p[permutation])
                prior_error.append(normalized_l1(pred_prior, target).cpu())
                shuffled_prior_error.append(
                    normalized_l1(pred_prior_shuffled, target).cpu()
                )
                prior_posterior_cosine.append(
                    F.cosine_similarity(p, q, dim=-1).cpu()
                )

                if hasattr(model.var_action, "prior_components"):
                    logits, _, _ = model.var_action.prior_components(prev)
                    probabilities = logits.softmax(-1)
                    prior_component_entropy.append(
                        (-(probabilities * probabilities.clamp_min(1e-12).log())
                         .sum(-1)).cpu()
                    )
                    prior_component_max.append(
                        probabilities.amax(-1).cpu()
                    )
                    prior_component_probabilities.append(
                        probabilities.cpu()
                    )

                # Use the model's actual plan-time prior. For a Gaussian this
                # matches the former direct reparameterization; for a mixture
                # it samples both the categorical component and its Gaussian.
                samples = model.var_action.sample_prior(
                    prev, k=args.prior_samples
                ).movedim(-2, 0)
                n_draws, n_transitions, d_action = samples.shape
                repeated_prev = prev.unsqueeze(0).expand(
                    n_draws, -1, -1
                ).reshape(n_draws * n_transitions, -1)
                sample_pred, _ = model.transition(
                    repeated_prev, samples.reshape(-1, d_action)
                )
                sample_pred = sample_pred.reshape(
                    n_draws, n_transitions, -1
                )
                shuffled_sample_pred, _ = model.transition(
                    repeated_prev, samples[:, permutation].reshape(
                        -1, d_action
                    )
                )
                shuffled_sample_pred = shuffled_sample_pred.reshape(
                    n_draws, n_transitions, -1
                )
                sample_error = normalized_l1(
                    sample_pred, target.unsqueeze(0)
                )
                shuffled_sample_error = normalized_l1(
                    shuffled_sample_pred, target.unsqueeze(0)
                )
                prior_best_error.append(sample_error.amin(0).cpu())
                shuffled_prior_best_error.append(
                    shuffled_sample_error.amin(0).cpu()
                )

            # Distributional calibration under the exact stochastic forward
            # pass used by the objective (sampled action and sampled target).
            used_pred = out.preds[bidx, tidx]
            used_logvar = out.extras["pred_logvar"][bidx, tidx]
            sampled_target = out.extras["target_sample"][bidx, tidx]
            residual = sampled_target - used_pred
            z = residual * torch.exp(-0.5 * used_logvar)
            predictive_nll.append(
                (0.5 * (used_logvar + z.pow(2))).mean(-1).cpu()
            )
            z_squared.append(z.pow(2).mean(-1).cpu())
            cover1.append((z.abs() <= 1.0).float().mean(-1).cpu())
            cover2.append((z.abs() <= 2.0).float().mean(-1).cpu())

            feature_tensors = (
                (("posterior", q), ("prior", p), ("sample", a), ("delta", delta))
                if action_mode == "latent"
                else (("observed_action", a), ("delta", delta))
            )
            for name, tensor in feature_tensors:
                features[name].append(tensor.cpu())
            own_error.append(normalized_l1(pred, target).cpu())
            shuffled_error.append(normalized_l1(pred_shuffled, target).cpu())
            mask = batch["step_mask"]
            labels["op"].append(batch["op"][mask].cpu())
            labels["necessary"].append(batch["necessary"][mask].cpu())
            labels["value"].append(batch["value"][mask].cpu())

    x = {name: torch.cat(parts) for name, parts in features.items()}
    y = {name: torch.cat(parts) for name, parts in labels.items()}
    results: dict[str, object] = {
        "checkpoint": args.ckpt,
        "action_source": action_mode,
        "n_transitions": int(y["op"].numel()),
        "matched_prediction_l1": float(torch.cat(own_error).mean()),
        "shuffled_action_prediction_l1": float(torch.cat(shuffled_error).mean()),
    }
    results["action_sensitivity_ratio"] = (
        results["shuffled_action_prediction_l1"]
        / max(results["matched_prediction_l1"], 1e-12)
    )
    if action_mode == "latent":
        mean_prior = float(torch.cat(prior_error).mean())
        mean_shuffled_prior = float(torch.cat(shuffled_prior_error).mean())
        best_prior = float(torch.cat(prior_best_error).mean())
        best_shuffled_prior = float(
            torch.cat(shuffled_prior_best_error).mean()
        )
        results["prior_mean_prediction_l1"] = mean_prior
        results["shuffled_prior_mean_prediction_l1"] = mean_shuffled_prior
        results["prior_mean_action_sensitivity_ratio"] = (
            mean_shuffled_prior / max(mean_prior, 1e-12)
        )
        results[f"prior_best_of_{args.prior_samples}_prediction_l1"] = best_prior
        results[
            f"shuffled_prior_best_of_{args.prior_samples}_prediction_l1"
        ] = best_shuffled_prior
        results[
            f"prior_best_of_{args.prior_samples}_action_sensitivity_ratio"
        ] = best_shuffled_prior / max(best_prior, 1e-12)
        results["prior_posterior_cosine"] = float(
            torch.cat(prior_posterior_cosine).mean()
        )
        if prior_component_entropy:
            entropy = torch.cat(prior_component_entropy)
            probabilities = torch.cat(prior_component_probabilities)
            results["prior_component_entropy"] = float(entropy.mean())
            results["prior_effective_components"] = float(
                entropy.mean().exp()
            )
            results["prior_max_component_probability"] = float(
                torch.cat(prior_component_max).mean()
            )
            results["prior_aggregate_component_probabilities"] = (
                probabilities.mean(0).tolist()
            )
    results["calibration"] = {
        "predictive_nll_without_constant": float(
            torch.cat(predictive_nll).mean()
        ),
        "mean_standardized_residual_squared": float(
            torch.cat(z_squared).mean()
        ),
        "coverage_1sigma": float(torch.cat(cover1).mean()),
        "coverage_2sigma": float(torch.cat(cover2).mean()),
    }

    task_classes = {"op": 4, "necessary": 2, "value": int(cfg.data.modulus)}
    probes = {}
    for source, feats in x.items():
        probes[source] = {
            task: ridge_probe_accuracy(feats, target, task_classes[task])
            for task, target in y.items()
        }
    results["linear_probe_accuracy"] = probes
    results["majority_accuracy"] = {
        task: float(torch.bincount(target).max() / target.numel())
        for task, target in y.items()
    }
    action_key = "posterior" if action_mode == "latent" else "observed_action"
    results[f"{action_key}_std"] = feature_std(x[action_key])
    results[f"{action_key}_effective_rank"] = effective_rank(
        x[action_key][:4096]
    )

    # Does each context-conditioned prior mean retrieve the posterior action
    # observed after the same transition?  Both codes contain the same unique
    # context, so this is a calibration diagnostic, not evidence that the
    # prior has recovered a reusable action identity.  The prior-prediction
    # and shuffled-prior fields above are the stronger pre-transition test.
    if action_mode == "latent":
        n = min(len(x["prior"]), 2048)
        prior = F.normalize(x["prior"][:n].float(), dim=-1)
        posterior = F.normalize(x["posterior"][:n].float(), dim=-1)
        sim = prior @ posterior.T
        order = sim.argsort(dim=1, descending=True)
        target = torch.arange(n).unsqueeze(1)
        rank = (order == target).nonzero()[:, 1] + 1
        results["prior_posterior_retrieval_top1"] = float(
            (rank == 1).float().mean()
        )
        results["prior_posterior_retrieval_mrr"] = float(
            (1.0 / rank.float()).mean()
        )

    destination = Path(args.out) if args.out else Path(args.ckpt).parent / "variational_probe.json"
    destination.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
