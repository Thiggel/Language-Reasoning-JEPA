"""Counterfactual dynamics, value, ranking, and support at macro scale."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from textjepa.objectives.base import Objective, latent_distance, masked_mean


class MacroCounterfactualDynamics(Objective):
    def __init__(self, kind: str = "smooth_l1", norm_targets: bool = True):
        super().__init__()
        self.kind = kind
        self.norm_targets = norm_targets

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "macro_cf_pred" not in out.extras:
            return out.preds.sum() * 0.0
        loss = latent_distance(
            out.extras["macro_cf_pred"],
            out.extras["macro_cf_target"],
            self.kind,
            self.norm_targets,
        )
        return masked_mean(loss, out.extras["macro_cf_valid"].float())


class HierarchyReachability(Objective):
    """Align high-level subgoals with endpoints imagined by the low model."""

    def __init__(self, kind: str = "smooth_l1", norm_targets: bool = True):
        super().__init__()
        self.kind = kind
        self.norm_targets = norm_targets

    def forward(self, out, batch: dict) -> torch.Tensor:
        losses = []
        factual_target = out.extras.get("hi_low_rollout_target")
        if factual_target is not None and out.hi_preds is not None:
            factual = latent_distance(
                out.hi_preds,
                factual_target.detach(),
                self.kind,
                self.norm_targets,
            )
            losses.append(masked_mean(factual, out.hi_mask.float()))
        cf_target = out.extras.get("macro_cf_low_target")
        if cf_target is not None:
            counterfactual = latent_distance(
                out.extras["macro_cf_pred"],
                cf_target.detach(),
                self.kind,
                self.norm_targets,
            )
            losses.append(masked_mean(
                counterfactual, out.extras["macro_cf_valid"].float()
            ))
        if not losses:
            return out.preds.sum() * 0.0
        return torch.stack(losses).mean()


class LowerHierarchyRollout(Objective):
    """Train the low model's K-step endpoint toward the encoded true state.

    This is the opposite side of :class:`HierarchyReachability`: instead of
    moving the high-level waypoint toward a frozen lower-model hallucination,
    it makes the recursively composed lower dynamics land on the state that
    the encoder assigns to the same executed action span.
    """

    def __init__(self, kind: str = "smooth_l1", norm_targets: bool = True):
        super().__init__()
        self.kind = kind
        self.norm_targets = norm_targets

    def forward(self, out, batch: dict) -> torch.Tensor:
        losses = []
        factual = out.extras.get("hi_low_rollout_target")
        if factual is not None and out.hi_targets is not None:
            distance = latent_distance(
                factual,
                out.hi_targets.detach(),
                self.kind,
                self.norm_targets,
            )
            losses.append(masked_mean(distance, out.hi_mask.float()))
        counterfactual = out.extras.get("macro_cf_low_target")
        target = out.extras.get("macro_cf_target")
        if counterfactual is not None and target is not None:
            distance = latent_distance(
                counterfactual,
                target.detach(),
                self.kind,
                self.norm_targets,
            )
            losses.append(masked_mean(
                distance, out.extras["macro_cf_valid"].float()
            ))
        if not losses:
            return out.preds.sum() * 0.0
        return torch.stack(losses).mean()


class MacroStateValue(Objective):
    """Distill exact remaining goal distance into V(predicted macro state)."""

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "macro_cf_state_value" not in out.extras:
            return out.preds.sum() * 0.0
        loss = F.smooth_l1_loss(
            out.extras["macro_cf_state_value"],
            out.extras["macro_cf_remaining"],
            reduction="none",
        )
        return masked_mean(loss, out.extras["macro_cf_valid"].float())


class MacroStateAdvantageRanking(Objective):
    """Rank predicted counterfactual macro outcomes by remaining cost."""

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "macro_cf_state_value" not in out.extras:
            return out.preds.sum() * 0.0
        value = out.extras["macro_cf_state_value"]
        remaining = out.extras["macro_cf_remaining"]
        valid = out.extras["macro_cf_valid"]
        better = remaining.unsqueeze(2) < remaining.unsqueeze(1)
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1) & better
        loss = F.relu(
            self.margin + value.unsqueeze(2) - value.unsqueeze(1)
        )
        return masked_mean(loss, pair_valid.float())


class MacroActionValue(Objective):
    """Distill Q(s,m) as remaining goal distance after the macro action."""

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "macro_cf_action_value" not in out.extras:
            return out.preds.sum() * 0.0
        loss = F.smooth_l1_loss(
            out.extras["macro_cf_action_value"],
            out.extras["macro_cf_remaining"],
            reduction="none",
        )
        return masked_mean(loss, out.extras["macro_cf_valid"].float())


class MacroAdvantageRanking(Objective):
    """Same-state pairwise ordering induced by exact macro advantages."""

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "macro_cf_action_value" not in out.extras:
            return out.preds.sum() * 0.0
        value = out.extras["macro_cf_action_value"]
        remaining = out.extras["macro_cf_remaining"]
        valid = out.extras["macro_cf_valid"]
        # i is better than j when it leaves fewer necessary steps. Since the
        # head predicts a cost, require Q_i + margin <= Q_j.
        better = remaining.unsqueeze(2) < remaining.unsqueeze(1)
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1) & better
        loss = F.relu(
            self.margin + value.unsqueeze(2) - value.unsqueeze(1)
        )
        return masked_mean(loss, pair_valid.float())


class MacroOODValueRanking(Objective):
    """Give perturbed, explicitly off-support macro codes a worse Q cost."""

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "macro_cf_perturbed_value" not in out.extras:
            return out.preds.sum() * 0.0
        positive = out.extras["macro_cf_action_value"].detach().unsqueeze(-1)
        negative = out.extras["macro_cf_perturbed_value"]
        loss = F.relu(self.margin + positive - negative)
        valid = out.extras["macro_cf_valid"].unsqueeze(-1).expand_as(loss)
        return masked_mean(loss, valid.float())


class SubgoalActionRanking(Objective):
    """Select the first primitive action that leads to a latent subgoal.

    Each valid counterfactual macro span supplies a target subgoal and its
    first intent action.  Other spans from the same state are negatives;
    duplicate first actions are treated as multiple positives.
    """

    def __init__(
        self, temperature: float = 1.0, predicted_weight: float = 1.0
    ):
        super().__init__()
        self.temperature = temperature
        self.predicted_weight = predicted_weight

    def _loss(self, cost: torch.Tensor, out) -> torch.Tensor:
        valid = out.extras["macro_cf_valid"]
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        positive = (
            out.extras["subgoal_action_positive"] & pair_valid
        )
        fallback = torch.zeros_like(pair_valid)
        fallback[:, :, 0] = ~valid
        safe_pair_valid = pair_valid | fallback
        safe_positive = positive | fallback
        logits = -cost / self.temperature
        denominator = torch.logsumexp(
            logits.masked_fill(~safe_pair_valid, -torch.inf), -1
        )
        numerator = torch.logsumexp(
            logits.masked_fill(~safe_positive, -torch.inf), -1
        )
        loss = denominator - numerator
        return masked_mean(loss, valid.float())

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "subgoal_action_cost" not in out.extras:
            return out.preds.sum() * 0.0
        losses = [self._loss(out.extras["subgoal_action_cost"], out)]
        if (
            self.predicted_weight
            and "subgoal_action_cost_pred" in out.extras
        ):
            losses.append(
                self.predicted_weight
                * self._loss(out.extras["subgoal_action_cost_pred"], out)
            )
        return torch.stack(losses).sum() / (
            1.0 + self.predicted_weight
        )


class MacroTop1Ranking(Objective):
    """Put probability mass on the best same-state macro alternatives."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "macro_cf_action_value" not in out.extras:
            return out.preds.sum() * 0.0
        value = out.extras["macro_cf_action_value"]
        remaining = out.extras["macro_cf_remaining"]
        valid = out.extras["macro_cf_valid"]
        anchor_valid = valid.any(-1)
        safe_valid = valid.clone()
        safe_valid[~anchor_valid, 0] = True
        large = torch.finfo(remaining.dtype).max
        best = remaining.masked_fill(
            ~safe_valid, large
        ).min(-1, keepdim=True).values
        optimal = safe_valid & (remaining == best)
        logits = -value / self.temperature
        denominator = torch.logsumexp(
            logits.masked_fill(~safe_valid, -torch.inf), -1
        )
        numerator = torch.logsumexp(
            logits.masked_fill(~optimal, -torch.inf), -1
        )
        loss = denominator - numerator
        return masked_mean(loss, anchor_valid.float())


def _discounted_prefix_cost(out, discount: float) -> torch.Tensor:
    prefix = out.extras["macro_cf_prefix_remaining"]
    horizon = prefix.shape[-1]
    weights = prefix.new_tensor([
        discount ** step for step in range(horizon)
    ])
    return (prefix * weights).sum(-1) / weights.sum().clamp_min(1e-8)


class MacroRecedingValue(Objective):
    """Terminal macro cost with an earlier-progress tie-break."""

    def __init__(self, discount: float = 0.5, tie_break_scale: float = 0.1):
        super().__init__()
        self.discount = discount
        self.tie_break_scale = tie_break_scale

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "macro_cf_prefix_remaining" not in out.extras:
            return out.preds.sum() * 0.0
        target = out.extras["macro_cf_remaining"] + (
            self.tie_break_scale * _discounted_prefix_cost(out, self.discount)
        )
        loss = F.smooth_l1_loss(
            out.extras["macro_cf_action_value"], target, reduction="none"
        )
        return masked_mean(loss, out.extras["macro_cf_valid"].float())


class MacroRecedingRanking(Objective):
    """Prefer terminal progress first and earlier prefix progress on ties."""

    def __init__(self, discount: float = 0.5, margin: float = 0.2):
        super().__init__()
        self.discount = discount
        self.margin = margin

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "macro_cf_prefix_remaining" not in out.extras:
            return out.preds.sum() * 0.0
        value = out.extras["macro_cf_action_value"]
        remaining = out.extras["macro_cf_remaining"]
        prefix = _discounted_prefix_cost(out, self.discount)
        valid = out.extras["macro_cf_valid"]
        terminal_better = remaining.unsqueeze(2) < remaining.unsqueeze(1)
        terminal_tie = remaining.unsqueeze(2) == remaining.unsqueeze(1)
        prefix_better = prefix.unsqueeze(2) < prefix.unsqueeze(1)
        better = terminal_better | (terminal_tie & prefix_better)
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1) & better
        loss = F.relu(
            self.margin + value.unsqueeze(2) - value.unsqueeze(1)
        )
        return masked_mean(loss, pair_valid.float())


class MacroSupport(Objective):
    """Separate task value from conditional action-manifold support."""

    def __init__(
        self, shuffled_weight: float = 1.0, perturbed_weight: float = 1.0
    ):
        super().__init__()
        self.shuffled_weight = shuffled_weight
        self.perturbed_weight = perturbed_weight

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "macro_cf_support_pos" not in out.extras:
            return out.preds.sum() * 0.0
        valid = out.extras["macro_cf_valid"].float()
        positive = F.binary_cross_entropy_with_logits(
            out.extras["macro_cf_support_pos"],
            torch.ones_like(out.extras["macro_cf_support_pos"]),
            reduction="none",
        )
        shuffled = F.binary_cross_entropy_with_logits(
            out.extras["macro_cf_support_shuffled"],
            torch.zeros_like(out.extras["macro_cf_support_shuffled"]),
            reduction="none",
        )
        perturbed = F.binary_cross_entropy_with_logits(
            out.extras["macro_cf_support_perturbed"],
            torch.zeros_like(out.extras["macro_cf_support_perturbed"]),
            reduction="none",
        )
        perturbed_valid = valid.unsqueeze(-1).expand_as(perturbed)
        return (
            masked_mean(positive, valid)
            + self.shuffled_weight * masked_mean(shuffled, valid)
            + self.perturbed_weight * masked_mean(
                perturbed, perturbed_valid
            )
        )


class ActionFeasibility(Objective):
    """Learn action availability without goal/relevance preference labels."""

    def __init__(self, positive_weight: float = 2.0):
        super().__init__()
        self.positive_weight = positive_weight

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "action_support_logits" not in out.extras:
            return out.preds.sum() * 0.0
        logits = out.extras["action_support_logits"]
        target = out.extras["action_support_target"].float()
        loss = -(
            self.positive_weight * target * F.logsigmoid(logits)
            + (1.0 - target) * F.logsigmoid(-logits)
        )
        return masked_mean(
            loss, out.extras["action_support_valid"].float()
        )
