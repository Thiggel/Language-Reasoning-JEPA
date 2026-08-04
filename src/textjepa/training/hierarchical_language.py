"""Stage-gated learner for hierarchical predictive-state language planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import nn

from textjepa.objectives.hierarchical_language import (
    EMAShrunkMahalanobis,
    SketchedIsotropicGaussianRegularizer,
    SquaredEuclideanMetric,
    commutation_loss,
    diagonal_gaussian_kl,
    dynamics_loss,
    masked_mean,
    listwise_value_loss,
    recursive_rollout_loss,
    supported_step_cost,
    vicreg_floor_and_covariance,
)


class ResearchStage(IntEnum):
    DATA_VALIDATION = 0
    TOKEN_JEPA = 1
    COUNTERFACTUAL_TOKEN = 1  # replay capability, not a separate gate
    FLAT_ORACLE_TOKEN = 2
    SENTENCE_JEPA = 3
    NESTED_SENTENCE = 3
    ORACLE_WAYPOINT = 4
    CROSS_LEVEL = 5
    DYNAMIC_COMMUTATION = 5
    MACRO_ACTION = 6
    ORACLE_HIGH_LEVEL = 7
    VALUE_DISTILLATION = 8
    FULL_HIERARCHY = 9
    CLOSED_LOOP_REANALYSIS = 10


@dataclass(frozen=True)
class DenseLossWeights:
    token_dynamics: float = 1.0
    sentence_dynamics: float = 1.0
    variance: float = 1.0
    covariance: float = 0.04
    sigreg: float = 1.0
    macro: float = 1.0
    commutation: float = 0.0
    token_rollout: float = 1.0

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.__dict__.values()):
            raise ValueError("loss weights must be nonnegative")


@dataclass(frozen=True)
class SparseRolloutSchedule:
    probability_n2: float = 0.25
    probability_n4: float = 0.05
    probability_n8: float = 0.01

    def __post_init__(self) -> None:
        probabilities = (
            self.probability_n2, self.probability_n4, self.probability_n8
        )
        if any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError("rollout probabilities must lie in [0, 1]")

    def sample_horizon(
        self, generator: torch.Generator | None = None
    ) -> int:
        draw = float(torch.rand((), generator=generator))
        if draw < self.probability_n8:
            return 8
        if draw < self.probability_n8 + self.probability_n4:
            return 4
        if draw < (
            self.probability_n8 + self.probability_n4 + self.probability_n2
        ):
            return 2
        return 1

    @staticmethod
    def truncate_bptt(horizon: int) -> int | None:
        if horizon <= 2:
            return None
        return 2


class HierarchicalLanguageLearner(nn.Module):
    """Compute only objectives admitted by the active scientific stage."""

    def __init__(
        self,
        model: nn.Module,
        stage: ResearchStage,
        weights: DenseLossWeights = DenseLossWeights(),
        *,
        covariance_momentum: float = 0.99,
        covariance_shrinkage: float = 0.1,
        macro_beta: float = 1.0,
        macro_free_bits: float = 0.0,
        dynamics_geometry: str = "mahalanobis",
        normalize_dynamics: bool = False,
        anti_collapse: str = "vicreg",
        sigreg_slices: int = 256,
        joint_token_sentence: bool = False,
    ):
        super().__init__()
        self.model = model
        self.stage = ResearchStage(stage)
        self.weights = weights
        self.macro_beta = float(macro_beta)
        self.macro_free_bits = float(macro_free_bits)
        self.dynamics_geometry = str(dynamics_geometry)
        self.normalize_dynamics = bool(normalize_dynamics)
        self.anti_collapse = str(anti_collapse)
        self.joint_token_sentence = bool(joint_token_sentence)
        if not 0 <= covariance_momentum < 1:
            raise ValueError("covariance momentum must be in [0, 1)")
        if not 0 <= covariance_shrinkage <= 1:
            raise ValueError("covariance shrinkage must be in [0, 1]")
        if macro_beta < 0 or macro_free_bits < 0:
            raise ValueError("macro beta and free bits must be nonnegative")
        if self.dynamics_geometry not in {"euclidean", "mahalanobis"}:
            raise ValueError("unknown dynamics geometry")
        if self.anti_collapse not in {"vicreg", "sigreg"}:
            raise ValueError("unknown anti-collapse regularizer")
        if weights.commutation and self.stage < ResearchStage.CROSS_LEVEL:
            raise ValueError("commutation is gated until cross-level prediction")
        config = model.config
        if self.stage >= ResearchStage.MACRO_ACTION and model.macro_actions is None:
            raise ValueError(
                "macro-action stages require enable_macro_actions=True"
            )
        if self.stage >= ResearchStage.VALUE_DISTILLATION and model.value is None:
            raise ValueError("value stages require enable_value=True")
        def metric(dimension: int) -> nn.Module:
            if self.dynamics_geometry == "mahalanobis":
                return EMAShrunkMahalanobis(
                    dimension,
                    covariance_momentum,
                    covariance_shrinkage,
                    normalized=self.normalize_dynamics,
                )
            return SquaredEuclideanMetric(
                dimension, normalized=self.normalize_dynamics
            )
        self.token_metric = metric(config.d_token)
        self.sentence_metric = metric(config.d_sentence)
        self.token_sigreg = SketchedIsotropicGaussianRegularizer(
            config.d_token, num_slices=sigreg_slices
        )
        self.sentence_sigreg = SketchedIsotropicGaussianRegularizer(
            config.d_sentence, num_slices=sigreg_slices
        )

    def _anti_collapse_losses(
        self,
        prefix: str,
        states: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.anti_collapse == "sigreg":
            regularizer = (
                self.token_sigreg if prefix == "token"
                else self.sentence_sigreg
            )
            return {f"{prefix}_sigreg": regularizer(states, mask)}
        variance, covariance = vicreg_floor_and_covariance(states, mask)
        return {
            f"{prefix}_variance": variance,
            f"{prefix}_covariance": covariance,
        }
    def forward(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        boundaries: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        prompt_len: torch.Tensor | None = None,
        solution_end: torch.Tensor | None = None,
        random_context_truncation: bool = True,
        token_rollout_start: torch.Tensor | None = None,
        token_rollout_actions: torch.Tensor | None = None,
        token_rollout_targets: torch.Tensor | None = None,
        token_rollout_mask: torch.Tensor | None = None,
        token_rollout_truncate_bptt: int | None = None,
        token_rollout_endpoint: torch.Tensor | None = None,
        macro_noise: torch.Tensor | None = None,
        value_successor_state: torch.Tensor | None = None,
        value_successor_context: torch.Tensor | None = None,
        value_teacher_cost: torch.Tensor | None = None,
        value_action_log_probability: torch.Tensor | None = None,
        value_action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.stage == ResearchStage.DATA_VALIDATION:
            # Stage 0 validates feature/index contracts in dense_forward but
            # admits no gradient-bearing JEPA objective.
            self.model.dense_forward(
                hidden, token_ids, boundaries,
                attention_mask=attention_mask,
                prompt_len=prompt_len,
                solution_end=solution_end,
                include_sentence=False,
                include_token_dynamics=False,
                random_context_truncation=False,
            )
            zero = hidden.new_zeros(())
            return zero, {"total": zero}
        include_sentence = (
            ResearchStage.SENTENCE_JEPA
            <= self.stage
            < ResearchStage.VALUE_DISTILLATION
        ) or self.stage == ResearchStage.CLOSED_LOOP_REANALYSIS
        include_token_dynamics = self.joint_token_sentence or self.stage in {
            ResearchStage.TOKEN_JEPA,
            ResearchStage.CLOSED_LOOP_REANALYSIS,
        }
        output = self.model.dense_forward(
            hidden, token_ids, boundaries,
            attention_mask=attention_mask,
            prompt_len=prompt_len,
            solution_end=solution_end,
            include_sentence=include_sentence,
            include_token_dynamics=include_token_dynamics,
            random_context_truncation=random_context_truncation,
        )
        with torch.no_grad():
            if include_token_dynamics:
                self.token_metric.update(
                    output["token_targets"], output["token_valid"]
                )
            if include_sentence:
                self.sentence_metric.update(
                    output["sentence_targets"], output["sentence_valid"]
                )
        losses: dict[str, torch.Tensor] = {}
        if include_token_dynamics:
            losses["token_dynamics"] = dynamics_loss(
                output["token_predictions"], output["token_targets"],
                output["token_valid"], self.token_metric,
            )
            losses.update(self._anti_collapse_losses(
                "token", output["token_states"], output["token_valid"]
            ))
        if token_rollout_actions is not None:
            if self.stage < ResearchStage.TOKEN_JEPA:
                raise ValueError("recursive rollout is gated by counterfactual stage")
            if token_rollout_targets is None or token_rollout_mask is None:
                raise ValueError("complete token rollout batch is required")
            rollout_start = (
                output["token_states"][:, 0]
                if token_rollout_start is None and "token_states" in output
                else token_rollout_start
            )
            if rollout_start is None:
                raise ValueError(
                    "token_rollout_start is required without dense P0 output"
                )
            losses["token_rollout"] = recursive_rollout_loss(
                self.model.token_predictor,
                rollout_start,
                token_rollout_actions,
                token_rollout_targets,
                token_rollout_mask,
                self.token_metric,
                truncate_bptt=token_rollout_truncate_bptt,
            )

        if include_sentence:
            losses["sentence_dynamics"] = dynamics_loss(
                output["sentence_predictions"], output["sentence_targets"],
                output["sentence_valid"], self.sentence_metric,
            )
            losses.update(self._anti_collapse_losses(
                "sentence", output["sentence_states"],
                output["sentence_valid"],
            ))

        if self.weights.commutation:
            if self.stage < ResearchStage.DYNAMIC_COMMUTATION:
                raise ValueError("commutation is gated until its ablation stage")
            if token_rollout_endpoint is None:
                if prompt_len is None or solution_end is None:
                    raise ValueError(
                        "commutation requires canonical prompt/solution bounds"
                    )
                token_rollout_endpoint = (
                    self._observed_sentence_token_rollout_endpoints(
                        hidden, token_ids, boundaries, prompt_len
                    )
                )
            losses["commutation"] = commutation_loss(
                token_rollout_endpoint,
                output["sentence_predictions"].detach(),
                output["sentence_valid"],
                self.model.e0_to_1,
                self.sentence_metric,
            )

        if self.stage >= ResearchStage.MACRO_ACTION:
            state = output["sentence_states"]
            observed = output["sentence_actions"]
            successor = output["sentence_targets"].detach()
            context = output["sentence_context"]
            task_prompt_len = (
                prompt_len
                if prompt_len is not None
                else boundaries[:, 0].clamp_min(1)
            )
            task = self.model.task_embedding(hidden, task_prompt_len)
            task = task[:, None].expand(
                *context.shape[:-1], task.shape[-1]
            )
            q_mean, q_logvar = self.model.macro_actions.posterior_params(
                state, context, observed, successor
            )
            p_mean, p_logvar = self.model.macro_actions.prior_params(
                state, context, task
            )
            sampled = self.model.macro_actions.reparameterize(
                q_mean, q_logvar, macro_noise
            )
            macro_prediction = self.model.sentence_predictor(
                state, sampled, output["sentence_valid"]
            )
            macro_dynamics = dynamics_loss(
                macro_prediction, successor, output["sentence_valid"],
                self.sentence_metric,
            )
            macro_kl = masked_mean(diagonal_gaussian_kl(
                q_mean, q_logvar, p_mean, p_logvar,
                free_bits=self.macro_free_bits,
            ), output["sentence_valid"])
            losses["macro"] = macro_dynamics + self.macro_beta * macro_kl

        if value_teacher_cost is not None:
            if self.stage < ResearchStage.VALUE_DISTILLATION:
                raise ValueError("value targets require value-distillation stage")
            required = (
                value_successor_state, value_successor_context,
                value_action_log_probability, value_action_mask,
            )
            if any(value is None for value in required):
                raise ValueError("complete value replay batch is required")
            task_prompt_len = (
                prompt_len
                if prompt_len is not None
                else boundaries[:, 0].clamp_min(1)
            )
            task = self.model.task_embedding(hidden, task_prompt_len)
            losses["value"] = value_distillation_loss(
                self.model,
                value_successor_state,
                value_successor_context,
                task,
                value_action_log_probability,
                value_teacher_cost,
                value_action_mask,
                step_cost=0.0,
                prior_weight=1.0,
                teacher_temperature=1.0,
                value_temperature=1.0,
            )

        total = hidden.sum() * 0
        if "token_dynamics" in losses:
            total = total + self.weights.token_dynamics * losses[
                "token_dynamics"
            ]
            if self.anti_collapse == "vicreg":
                total = (
                    total
                    + self.weights.variance * losses["token_variance"]
                    + self.weights.covariance * losses["token_covariance"]
                )
            else:
                total = total + self.weights.sigreg * losses["token_sigreg"]
        if "sentence_dynamics" in losses:
            total = total + self.weights.sentence_dynamics * losses[
                "sentence_dynamics"
            ]
            if self.anti_collapse == "vicreg":
                total = (
                    total
                    + self.weights.variance * losses["sentence_variance"]
                    + self.weights.covariance * losses["sentence_covariance"]
                )
            else:
                total = total + self.weights.sigreg * losses[
                    "sentence_sigreg"
                ]
        if "macro" in losses:
            total = total + self.weights.macro * losses["macro"]
        if "commutation" in losses:
            total = total + self.weights.commutation * losses["commutation"]
        if "token_rollout" in losses:
            total = total + self.weights.token_rollout * losses["token_rollout"]
        if "value" in losses:
            total = total + losses["value"]
        losses["total"] = total
        return total, losses

    def _observed_sentence_token_rollout_endpoints(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        boundaries: torch.Tensor,
        prompt_len: torch.Tensor,
    ) -> torch.Tensor:
        """Roll P0 through each observed sentence with its real root cache."""
        token = self.model.encode_token(hidden)
        batch, steps = boundaries.shape[0], boundaries.shape[1] - 1
        endpoint = token.new_zeros(batch, steps, token.shape[-1])
        for row in range(batch):
            valid_boundaries = boundaries[row][boundaries[row] >= 0]
            for step in range(len(valid_boundaries) - 1):
                start = int(valid_boundaries[step])
                end = int(valid_boundaries[step + 1])
                history_start = max(
                    int(prompt_len[row]),
                    start - self.model.config.token_context + 1,
                )
                state_history = token[
                    row, history_start - 1:start
                ][None]
                action_history_ids = token_ids[
                    row, history_start:start
                ][None]
                action_history = self.model.token_action(
                    action_history_ids
                )
                actions = self.model.token_action(
                    token_ids[row, start:end][None]
                )
                rollout, _ = self.model.p0.rollout(
                    state_history[:, -1], actions,
                    state_history=state_history,
                    action_history=action_history,
                )
                endpoint[row, step] = rollout[0, -1]
        return endpoint

    def counterfactual_loss(
        self,
        root_hidden: torch.Tensor,
        suffix_hidden: torch.Tensor,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        *,
        sentence_eligible: torch.Tensor | None = None,
        rollout_horizon: int = 1,
        rollout_truncate_bptt: int | None = None,
        root_token_history_hidden: torch.Tensor | None = None,
        root_token_history_action_ids: torch.Tensor | None = None,
        root_token_history_lengths: torch.Tensor | None = None,
        root_sentence_history_hidden: torch.Tensor | None = None,
        root_sentence_history_lengths: torch.Tensor | None = None,
        root_sentence_history_span_ids: torch.Tensor | None = None,
        root_sentence_history_span_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Train P0 on every generated token and P1 only on true boundaries."""
        if self.stage < ResearchStage.TOKEN_JEPA:
            raise ValueError("counterfactual replay requires its admitted stage")
        if root_hidden.ndim != 2 or suffix_hidden.ndim != 3:
            raise ValueError("counterfactual hidden-state shapes are invalid")
        if token_ids.shape != suffix_hidden.shape[:2] or lengths.shape != (
            len(token_ids),
        ):
            raise ValueError("counterfactual tensors do not align")
        width = token_ids.shape[1]
        valid = torch.arange(
            width, device=token_ids.device
        )[None] < lengths[:, None]
        suffix = self.model.e0(suffix_hidden)
        with torch.no_grad():
            target = self.model.e0_target(suffix_hidden)
        actions = self.model.token_action(token_ids)
        if root_token_history_hidden is None:
            root_token_history_hidden = root_hidden[:, None]
            root_token_history_action_ids = token_ids[:, :0]
            root_token_history_lengths = torch.ones(
                len(token_ids), dtype=torch.long, device=token_ids.device
            )
        if root_token_history_action_ids is None or (
            root_token_history_lengths is None
        ):
            raise ValueError("complete token predictor history is required")
        prediction = suffix.new_zeros(suffix.shape)
        token_histories: list[tuple[torch.Tensor, torch.Tensor]] = []
        for row in range(len(token_ids)):
            history_length = int(root_token_history_lengths[row])
            branch_length = int(lengths[row])
            history_state = self.model.e0(
                root_token_history_hidden[row, :history_length]
            )
            history_action = self.model.token_action(
                root_token_history_action_ids[row, :history_length - 1]
            )
            branch_states = torch.cat([
                history_state, suffix[row, :branch_length - 1]
            ], 0)
            branch_actions = torch.cat([
                history_action, actions[row, :branch_length]
            ], 0)
            branch_prediction = self.model.p0(
                branch_states[None], branch_actions[None]
            )[0, history_length - 1:]
            prediction[row, :branch_length] = branch_prediction
            token_histories.append((history_state, history_action))
        token_loss = dynamics_loss(
            prediction, target, valid, self.token_metric
        )
        losses = {"counterfactual_token": token_loss}
        total = self.weights.token_dynamics * token_loss
        replay_regularization = self._anti_collapse_losses(
            "token", suffix, valid
        )
        replay_regularization = {
            f"counterfactual_{name}": value
            for name, value in replay_regularization.items()
        }
        losses.update(replay_regularization)
        if self.anti_collapse == "vicreg":
            total = (
                total
                + self.weights.variance
                * losses["counterfactual_token_variance"]
                + self.weights.covariance
                * losses["counterfactual_token_covariance"]
            )
        else:
            total = total + self.weights.sigreg * losses[
                "counterfactual_token_sigreg"
            ]
        if rollout_horizon not in {1, 2, 4, 8}:
            raise ValueError("counterfactual rollout horizon must be 1/2/4/8")
        if rollout_horizon > 1:
            eligible = (lengths >= rollout_horizon).nonzero().flatten()
            if len(eligible):
                rollout_terms = []
                for row_tensor in eligible:
                    row = int(row_tensor)
                    history_state, history_action = token_histories[row]
                    rollout_actions = actions[
                        row:row + 1, :rollout_horizon
                    ]
                    if rollout_truncate_bptt is None:
                        rollout, _ = self.model.p0.rollout(
                            history_state[-1], rollout_actions,
                            state_history=history_state[None],
                            action_history=history_action[None],
                        )
                    else:
                        if rollout_truncate_bptt < 1:
                            raise ValueError(
                                "rollout BPTT truncation must be positive"
                            )
                        state_history = history_state[None]
                        action_history = history_action[None]
                        predictions = []
                        for step in range(rollout_horizon):
                            segment = rollout_actions[:, step:step + 1]
                            one, _ = self.model.p0.rollout(
                                state_history[:, -1], segment,
                                state_history=state_history,
                                action_history=action_history,
                            )
                            current = one[:, -1]
                            predictions.append(current)
                            state_history = torch.cat(
                                [state_history, current[:, None]], 1
                            )
                            action_history = torch.cat(
                                [action_history, segment], 1
                            )
                            if (
                                (step + 1) % rollout_truncate_bptt == 0
                                and step + 1 < rollout_horizon
                            ):
                                state_history = state_history.detach()
                                action_history = action_history.detach()
                        rollout = torch.stack(predictions, 1)
                    rollout_terms.append(self.token_metric(
                        rollout[0], target[row, :rollout_horizon]
                    ).mean())
                rollout_loss = torch.stack(rollout_terms).mean()
                losses["counterfactual_rollout"] = rollout_loss
                total = total + self.weights.token_rollout * rollout_loss
        if sentence_eligible is not None and bool(sentence_eligible.any()):
            if self.stage < ResearchStage.SENTENCE_JEPA:
                return total, losses
            indices = sentence_eligible.nonzero().flatten()
            rows = suffix_hidden[indices]
            row_token_ids = token_ids[indices]
            row_valid = valid[indices]
            action = self.model.a1(row_token_ids, row_valid)
            with torch.no_grad():
                last = lengths[indices] - 1
                target_state = self.model.encode_sentence(
                    rows[torch.arange(len(indices), device=rows.device), last],
                    target=True,
                )
            sentence_predictions = []
            for local, row_tensor in enumerate(indices):
                row = int(row_tensor)
                if root_sentence_history_hidden is None:
                    history_hidden = root_hidden[row:row + 1]
                    prior_action = action[local:local, :]
                else:
                    if root_sentence_history_lengths is None or (
                        root_sentence_history_span_ids is None
                    ) or root_sentence_history_span_mask is None:
                        raise ValueError(
                            "complete sentence predictor history is required"
                        )
                    history_length = int(root_sentence_history_lengths[row])
                    history_hidden = root_sentence_history_hidden[
                        row, :history_length
                    ]
                    if history_length == 1:
                        # A branch rooted at the initial solution boundary has
                        # one state and no preceding sentence actions.  Avoid
                        # sending a zero-sized batch through TransformerEncoder,
                        # which PyTorch cannot reshape in multi-head attention.
                        prior_action = action.new_empty(0, action.shape[-1])
                    else:
                        prior_action = self.model.a1(
                            root_sentence_history_span_ids[
                                row, :history_length - 1
                            ],
                            root_sentence_history_span_mask[
                                row, :history_length - 1
                            ],
                        )
                history_state = self.model.encode_sentence(history_hidden)
                all_actions = torch.cat(
                    [prior_action, action[local:local + 1]], 0
                )
                predicted = self.model.p1(
                    history_state[None], all_actions[None]
                )
                sentence_predictions.append(predicted[0, -1])
            sentence_prediction = torch.stack(sentence_predictions)
            sentence_loss = self.sentence_metric(
                sentence_prediction, target_state
            ).mean()
            losses["counterfactual_sentence"] = sentence_loss
            total = total + self.weights.sentence_dynamics * sentence_loss
            last = lengths[indices] - 1
            achieved_online = self.model.encode_sentence(
                rows[torch.arange(len(indices), device=rows.device), last]
            )
            root_online = self.model.encode_sentence(root_hidden[indices])
            sentence_states = torch.stack([root_online, achieved_online], 1)
            sentence_mask = torch.ones(
                sentence_states.shape[:2],
                dtype=torch.bool,
                device=sentence_states.device,
            )
            replay_sentence_regularization = self._anti_collapse_losses(
                "sentence", sentence_states, sentence_mask
            )
            replay_sentence_regularization = {
                f"counterfactual_{name}": value
                for name, value in replay_sentence_regularization.items()
            }
            losses.update(replay_sentence_regularization)
            if self.anti_collapse == "vicreg":
                total = (
                    total
                    + self.weights.variance
                    * losses["counterfactual_sentence_variance"]
                    + self.weights.covariance
                    * losses["counterfactual_sentence_covariance"]
                )
            else:
                total = total + self.weights.sigreg * losses[
                    "counterfactual_sentence_sigreg"
                ]
        losses["total"] = total
        return total, losses


def value_distillation_loss(
    model: nn.Module,
    successor_state: torch.Tensor,
    successor_context: torch.Tensor,
    task: torch.Tensor,
    first_action_log_probability: torch.Tensor,
    teacher_cost: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    step_cost: float,
    prior_weight: float,
    teacher_temperature: float,
    value_temperature: float,
) -> torch.Tensor:
    """Rank first actions using offline oracle-search costs.

    Offline continuation sample/depth axes have already been reduced into
    ``teacher_cost``.  Neither those depths nor a remaining budget enter the
    value model.
    """
    if model.value is None:
        raise ValueError("value distillation requires enable_value=True")
    continuation = model.value(successor_state, successor_context, task)
    predicted = supported_step_cost(
        first_action_log_probability, step_cost, prior_weight
    ) + continuation
    return listwise_value_loss(
        teacher_cost, predicted, action_mask,
        teacher_temperature, value_temperature,
    )
