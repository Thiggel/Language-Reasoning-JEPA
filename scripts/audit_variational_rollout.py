"""Audit open-loop uncertainty growth in observed-action variational JEPA.

The true intent sequence is held fixed. Starting at the encoded prompt state,
the learned Gaussian transition is sampled recursively. At each horizon we
compare the Monte Carlo predictive distribution with the target encoder's
state under the true rendered history. This separates one-step calibration
from accumulated open-loop bias and spread.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from textjepa.training.trainer import to_device
from textjepa.utils.checkpoint import build_dataset, collate_for, load_run
from textjepa.utils.seed import seed_everything


def normalized_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.layer_norm(x, x.shape[-1:])
    y = F.layer_norm(y, y.shape[-1:])
    return (x - y).abs().mean(-1)


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def calibration_metrics(z: torch.Tensor, temperature: float) -> dict[str, float]:
    """Summarize a frozen scalar spread correction on standardized residuals."""
    scaled = z / max(float(temperature), 1e-6)
    return {
        "z2_raw": float(z.square().mean()),
        "z2_calibrated": float(scaled.square().mean()),
        "coverage_1sigma_raw": float((z.abs() <= 1).float().mean()),
        "coverage_1sigma_calibrated": float(
            (scaled.abs() <= 1).float().mean()
        ),
        "coverage_2sigma_raw": float((z.abs() <= 2).float().mean()),
        "coverage_2sigma_calibrated": float(
            (scaled.abs() <= 2).float().mean()
        ),
    }


@torch.no_grad()
def audit(args) -> dict:
    seed_everything(args.seed)
    device = torch.device(args.device)
    model, vocab, cfg = load_run(args.ckpt, args.device)
    if model.__class__.__name__ != "DiscourseVJEPA":
        raise TypeError("rollout audit requires DiscourseVJEPA")
    if getattr(model, "action_mode", "latent") == "latent":
        raise ValueError("rollout audit requires an observed action mode")
    dataset = build_dataset(
        cfg, vocab, split=args.split,
        size=min(args.n_examples, int(cfg.data.get(f"{args.split}_size"))),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate_for(cfg), pad_id=vocab.pad_id),
    )
    by_horizon: dict[int, dict[str, list[float]]] = {}
    calibration_z: dict[int, dict[str, list[torch.Tensor]]] = {}
    calibration_cutoff = int(len(dataset) * args.calibration_fraction)
    external_source = None
    external_temperatures: dict[int, float] = {}
    if args.temperature_source:
        external_source = json.loads(Path(args.temperature_source).read_text())
        for horizon, row in external_source.get("by_horizon", {}).items():
            temperature = row.get("variance_temperature")
            if temperature is not None:
                external_temperatures[int(horizon)] = float(temperature)
        if not external_temperatures:
            raise ValueError(
                "temperature source contains no fitted variance temperatures"
            )
    n_seen = 0
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch)
        B, T, D = out.step_states_tgt.shape
        draws = args.rollout_samples
        current = out.s0.unsqueeze(0).expand(draws, -1, -1).clone()
        for t in range(T):
            action = out.actions[:, t]
            repeated_action = action.unsqueeze(0).expand(draws, -1, -1)
            mean, logvar = model.transition(
                current.reshape(draws * B, D),
                repeated_action.reshape(draws * B, -1),
            )
            mean = mean.reshape(draws, B, D)
            logvar = logvar.reshape(draws, B, D)
            current = mean + torch.randn_like(mean) * (0.5 * logvar).exp()

            valid = out.step_mask[:, t]
            if not valid.any():
                continue
            target = out.step_states_tgt[valid, t]
            predictive_mean = current[:, valid].mean(0)
            predictive_var = current[:, valid].var(0, unbiased=False).clamp_min(1e-6)
            predictive_std = predictive_var.sqrt()
            residual_z = (target - predictive_mean) / predictive_std
            global_index = torch.arange(
                n_seen, n_seen + B, device=device
            )[valid]
            one_step_mean = out.preds[valid, t]
            one_step_std = (0.5 * out.extras["pred_logvar"][valid, t]).exp()

            row = by_horizon.setdefault(t + 1, {
                "open_loop_l1": [],
                "open_loop_std": [],
                "open_loop_z2": [],
                "open_loop_coverage_1sigma": [],
                "open_loop_coverage_2sigma": [],
                "teacher_forced_l1": [],
                "teacher_forced_std": [],
            })
            row["open_loop_l1"].extend(
                normalized_l1(predictive_mean, target).cpu().tolist()
            )
            row["open_loop_std"].extend(
                predictive_std.mean(-1).cpu().tolist()
            )
            row["open_loop_z2"].extend(
                residual_z.square().mean(-1).cpu().tolist()
            )
            row["open_loop_coverage_1sigma"].extend(
                (residual_z.abs() <= 1).float().mean(-1).cpu().tolist()
            )
            row["open_loop_coverage_2sigma"].extend(
                (residual_z.abs() <= 2).float().mean(-1).cpu().tolist()
            )
            row["teacher_forced_l1"].extend(
                normalized_l1(one_step_mean, target).cpu().tolist()
            )
            row["teacher_forced_std"].extend(
                one_step_std.mean(-1).cpu().tolist()
            )
            split = calibration_z.setdefault(
                t + 1, {"all": [], "fit": [], "evaluation": []}
            )
            split["all"].append(residual_z.detach().cpu())
            fit = global_index < calibration_cutoff
            if fit.any():
                split["fit"].append(residual_z[fit].detach().cpu())
            if (~fit).any():
                split["evaluation"].append(
                    residual_z[~fit].detach().cpu()
                )
        n_seen += B

    summary = {}
    for horizon, metrics in sorted(by_horizon.items()):
        row = {
            "n": len(metrics["open_loop_l1"]),
            **{name: average(values) for name, values in metrics.items()},
        }
        split = calibration_z[horizon]
        if split["fit"] and split["evaluation"]:
            fit_z = torch.cat(split["fit"], dim=0)
            eval_z = torch.cat(split["evaluation"], dim=0)
            temperature = fit_z.square().mean().sqrt().clamp_min(1e-6)
            heldout = calibration_metrics(eval_z, float(temperature))
            row.update({
                "temperature_fit_n": int(fit_z.shape[0]),
                "temperature_evaluation_n": int(eval_z.shape[0]),
                "variance_temperature": float(temperature),
                **{f"evaluation_{key}": value for key, value in heldout.items()},
            })
        if horizon in external_temperatures:
            all_z = torch.cat(split["all"], dim=0)
            external_temperature = external_temperatures[horizon]
            transfer = calibration_metrics(all_z, external_temperature)
            row.update({
                "transfer_n": int(all_z.shape[0]),
                "external_variance_temperature": external_temperature,
                **{f"transfer_{key}": value for key, value in transfer.items()},
            })
        summary[str(horizon)] = row
    result = {
        "checkpoint": args.ckpt,
        "split": args.split,
        "examples": n_seen,
        "rollout_samples": args.rollout_samples,
        "action_mode": model.action_mode,
        "target_mode": model.target_mode,
        "temperature_calibration": {
            "fit_fraction": args.calibration_fraction,
            "fit_prefix_examples": calibration_cutoff,
            "evaluation_suffix_examples": len(dataset) - calibration_cutoff,
        },
        "by_horizon": summary,
    }
    if external_source is not None:
        result["temperature_transfer"] = {
            "source": str(args.temperature_source),
            "source_checkpoint": external_source.get("checkpoint"),
            "source_split": external_source.get("split"),
            "evaluation_split": args.split,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--n-examples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--rollout-samples", type=int, default=16)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument(
        "--temperature-source",
        help=(
            "JSON audit containing validation-fitted variance temperatures; "
            "apply them unchanged to every example in this audit"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()
    if not 0 < args.calibration_fraction < 1:
        parser.error("--calibration-fraction must lie strictly between 0 and 1")
    result = audit(args)
    destination = (
        Path(args.out) if args.out else
        Path(args.ckpt).parent / "variational_rollout.json"
    )
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
