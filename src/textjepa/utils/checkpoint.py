"""Checkpoint loading and dataset construction shared by all scripts."""

from __future__ import annotations

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from textjepa.data.edits.dataset import EditDataset, collate_edits
from textjepa.data.igsm.dataset import IGSMDataset, build_vocab, collate


def apply_eval_data_overrides(run_cfg, eval_cfg) -> None:
    """Apply explicit post-hoc dataset shifts to a loaded run config.

    Training checkpoints remain self-describing by default. These overrides
    are opt-in and intended for controlled OOD evaluation only.
    """
    mappings = {
        "eval_steps_range": "steps_range",
        "eval_n_vars_range": "n_vars_range",
        "eval_leaf_prob": "leaf_prob",
        "eval_sample_max_tries": "sample_max_tries",
        "eval_strict_steps_range": "strict_steps_range",
        # Faithful (official-iGSM) generator knobs: the paper's OOD design
        # shifts the operation count, not our stylized step/variable counts.
        "eval_op_range": "op_range",
        "eval_max_op": "max_op",
        "eval_max_edge": "max_edge",
        "eval_necessary_range": "necessary_range",
    }
    for source, destination in mappings.items():
        value = eval_cfg.get(source)
        if value is not None:
            run_cfg.data[destination] = value


def build_vocab_for_config(cfg):
    """Build a vocabulary without inspecting validation or test text."""
    name = cfg.data.get("name", "igsm")
    if name == "observed_action":
        from textjepa.data.observed_action import (
            build_observed_action_vocab,
            load_observed_action_jsonl,
        )

        episodes = load_observed_action_jsonl(
            cfg.data.train_path,
            expected_domain=cfg.data.get("domain"),
        )
        return build_observed_action_vocab(episodes)
    if name == "igsm_real_token_edit":
        from textjepa.data.faithful_token_edits import (
            faithful_replacement_vocab, faithful_token_edit_vocab,
        )

        return (
            faithful_replacement_vocab()
            if cfg.data.get("replacement_only_vocab", False)
            else faithful_token_edit_vocab()
        )
    if name == "igsm_real":
        from textjepa.data.faithful import cached_faithful_vocab

        return cached_faithful_vocab()
    return build_vocab(cfg.data.modulus)


def _migrate_legacy_state_dict(state: dict[str, torch.Tensor]):
    """Adapt checkpoints created before ``MacroActionModel`` wrapped its encoder.

    Historical transformer macro encoders lived directly at
    ``core.macro_encoder.*``.  The current module keeps the identical encoder
    under ``core.macro_encoder.encoder.*`` and adds a conditional prior.  Only
    the historical encoder tensors are renamed; the newly added prior remains
    randomly initialized and is reported through the ordinary missing-key
    path.
    """
    legacy_marker = "core.macro_encoder.cls"
    current_marker = "core.macro_encoder.encoder.cls"
    if legacy_marker not in state or current_marker in state:
        return state
    prefix = "core.macro_encoder."
    migrated = {}
    for name, value in state.items():
        if name.startswith(prefix):
            name = prefix + "encoder." + name[len(prefix):]
        migrated[name] = value
    return migrated


def load_run(ckpt_path: str, device: str = "cuda:0", random_init: bool = False):
    """Returns (model, vocab, cfg) from a training checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    vocab = build_vocab_for_config(cfg)
    model = instantiate(cfg.model, vocab_size=len(vocab), pad_id=vocab.pad_id)
    if not random_init:
        state = _migrate_legacy_state_dict(ckpt["model"])
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected checkpoint keys: {unexpected}")
        if missing:
            print(f"note: initializing modules added after this run: {missing}")
    return model.to(device).eval(), vocab, cfg


def build_dataset(cfg, vocab, split: str = "val", size: int | None = None):
    d = cfg.data
    if d.get("name", "igsm") == "observed_action":
        from textjepa.data.observed_action import (
            ObservedActionDataset,
            load_observed_action_jsonl,
        )

        path = d.get(f"{split}_path")
        if path is None:
            raise ValueError(f"missing observed-action {split}_path")
        # Compiled domains fix several knobs at compile time.  Fail loudly on
        # any non-default value rather than silently dropping it: a silently
        # dropped data flag has already cost this project two invalid screens.
        # Structurally fixed by compilation, hence deliberately not runtime
        # knobs here: ``geo_rank_rollouts`` (compiled data stores exactly the
        # recorded teacher continuation per candidate) and
        # ``geo_rank_rollout_for_h1`` (the recorded rollout always includes
        # the immediate consequence, i.e. the ``true`` semantics), and
        # ``geo_rank_beam_width`` (a property of the teacher that produced the
        # recorded continuation, recorded in the domain MANIFEST).
        for key, default in (
            ("geo_rank_policy", "random"),
            ("invalid_action_mode", "noop"),
            ("geo_rank_factual_only", False),
            ("all_action_supervision", False),
            ("macro_alt_k", 0),
            ("n_alt", 0),
        ):
            value = d.get(key, default)
            if value != default:
                raise NotImplementedError(
                    f"observed-action data ignores data.{key}={value!r}; "
                    "recompile the domain instead of setting it at train time"
                )

        episodes = load_observed_action_jsonl(
            path, expected_domain=d.get("domain")
        )
        # ``<split>_size`` caps how many recorded episodes are used, which is
        # how tiny admission cells subset a full compiled corpus.  ``null``
        # keeps every episode in the file.
        if size is None:
            size = d.get(f"{split}_size", None)
        if size is not None:
            episodes = episodes[:int(size)]
        return ObservedActionDataset(
            episodes,
            vocab,
            geo_rank_k=d.get("geo_rank_k", 0),
            geo_rank_horizon=d.get("geo_rank_horizon", 1),
            geo_rank_horizons=d.get("geo_rank_horizons", None),
            geo_rank_candidate_interface=d.get(
                "geo_rank_candidate_interface", "compiled"
            ),
            geo_rank_feasible_k=d.get("geo_rank_feasible_k", None),
            geo_rank_invalid_k=d.get("geo_rank_invalid_k", None),
            dense_geo_anchors=(
                split == "train" and d.get("dense_geo_anchors", False)
            ),
            shuffle_actions=(
                split == "train" and d.get("shuffle_actions", False)
            ),
            seed=d.get(f"{split}_seed", 0),
        )
    igsm_kwargs = dict(
        modulus=d.modulus,
        n_vars_range=tuple(d.n_vars_range),
        leaf_prob=d.leaf_prob,
        steps_range=tuple(d.steps_range),
        distractor_prob=d.distractor_prob,
        max_distractors=d.max_distractors,
        # .get: configs stored in older checkpoints lack these keys
        shuffle_actions=d.get("shuffle_actions", False),
        n_alt=d.get("n_alt", 0),
        geo_rank_k=d.get("geo_rank_k", 0),
        geo_rank_factual_only=d.get("geo_rank_factual_only", False),
        geo_rank_horizon=d.get("geo_rank_horizon", 1),
        geo_rank_horizons=d.get("geo_rank_horizons", None),
        geo_rank_rollouts=d.get("geo_rank_rollouts", 1),
        geo_rank_rollout_for_h1=d.get("geo_rank_rollout_for_h1", False),
        geo_rank_policy=d.get("geo_rank_policy", "random"),
        geo_rank_beam_width=d.get("geo_rank_beam_width", 1),
        geo_rank_candidate_interface=d.get(
            "geo_rank_candidate_interface", "feasible_menu"
        ),
        geo_rank_feasible_k=d.get("geo_rank_feasible_k", None),
        geo_rank_invalid_k=d.get("geo_rank_invalid_k", None),
        invalid_action_mode=d.get("invalid_action_mode", "noop"),
        macro_alt_k=d.get("macro_alt_k", 0),
        macro_alt_horizon=d.get("macro_alt_horizon", 3),
        all_action_supervision=d.get("all_action_supervision", False),
        sample_max_tries=d.get("sample_max_tries", 50),
        strict_steps_range=d.get("strict_steps_range", False),
    )
    if split == "train":
        default_size, seed = d.train_size, d.train_seed
    elif split == "val":
        default_size, seed = d.val_size, d.val_seed
    elif split == "test":
        default_size = d.get("test_size", d.val_size)
        seed = d.get("test_seed", int(d.val_seed) + 1)
    else:
        raise ValueError(f"unknown dataset split: {split}")
    size = size or default_size
    if d.get("name", "igsm") == "igsm_real_token_edit":
        from textjepa.data.faithful_token_edits import FaithfulTokenEditDataset

        dataset = FaithfulTokenEditDataset(
            vocab, size=size, seed=seed, max_op=d.max_op,
            max_edge=d.max_edge, op_range=tuple(d.op_range),
            min_edits=d.min_edits, max_edits=d.max_edits,
            counterfactual_k=d.get("counterfactual_k", 0),
            proposal_pool_k=d.get("proposal_pool_k", 0),
            proposal_token_pool=d.get(
                "proposal_token_pool", "current_buffer"
            ),
            counterfactual_source=d.get(
                "counterfactual_source", "uniform_local"
            ),
            corruption_mode=(
                "mixed" if split != "train" and
                d.get("corruption_mode", "mixed") == "curriculum"
                else d.get("corruption_mode", "mixed")
            ),
            curriculum_epochs=d.get("curriculum_epochs", 3),
            fresh_per_epoch=d.get("fresh_per_epoch", False),
            gar_teacher=d.get("gar_teacher", "latent_distance"),
            trajectory_variants=(
                d.get("trajectory_variants", 1) if split == "train"
                else d.get("eval_trajectory_variants", 1)
            ),
            refinement_probability=d.get("refinement_probability", 0.25),
            sample_transition=d.get("sample_transition", False),
            content_only_actions=d.get("content_only_actions", False),
        )
        replay_path = d.get("replay_path")
        replay_fraction = float(d.get("replay_fraction", 0.0))
        if split == "train" and replay_path and replay_fraction > 0:
            from textjepa.data.token_edit_replay import (
                FrozenPolicyReplayDataset,
                MixedReplayTokenEditDataset,
            )

            replay = FrozenPolicyReplayDataset(
                replay_path, vocab,
                proposal_pool_k=d.get("proposal_pool_k", 0),
                proposal_token_pool=d.get(
                    "proposal_token_pool", "prompt_plus_current"
                ),
                gar_teacher=d.get("gar_teacher", "latent_distance"),
                max_depth=d.get("replay_max_depth"),
                seed=d.get("replay_proposal_seed", seed),
            )
            dataset = MixedReplayTokenEditDataset(
                dataset, replay, replay_fraction
            )
        return dataset
    if d.get("name", "igsm") == "igsm_real":
        from textjepa.data.faithful import FaithfulDataset

        return FaithfulDataset(
            vocab, size=size, seed=seed,
            max_op=d.max_op, max_edge=d.max_edge,
            op_range=tuple(d.op_range),
            necessary_range=tuple(d.get("necessary_range", (None, None))
                                  or (None, None)),
            distractor_prob=d.distractor_prob,
            max_distractors=d.max_distractors,
            n_alt=d.get("n_alt", 0),
            geo_rank_k=d.get("geo_rank_k", 0),
            geo_rank_horizon=d.get("geo_rank_horizon", 1),
            geo_rank_horizons=d.get("geo_rank_horizons", None),
            geo_rank_rollout_for_h1=d.get("geo_rank_rollout_for_h1", False),
            geo_rank_candidate_interface=d.get(
                "geo_rank_candidate_interface", "compiled"
            ),
            geo_rank_factual_only=d.get("geo_rank_factual_only", False),
            geo_rank_feasible_k=d.get("geo_rank_feasible_k", None),
            geo_rank_invalid_k=d.get("geo_rank_invalid_k", None),
            invalid_action_mode=d.get("invalid_action_mode", "noop"),
            geo_rank_rollouts=d.get("geo_rank_rollouts", 1),
            geo_rank_policy=d.get("geo_rank_policy", "random"),
            geo_rank_beam_width=d.get("geo_rank_beam_width", 1),
            invalid_counterfactual_k=d.get("invalid_counterfactual_k", 0),
            macro_alt_k=d.get("macro_alt_k", 0),
            macro_alt_horizon=d.get("macro_alt_horizon", 3),
            all_action_supervision=d.get("all_action_supervision", False),
            shuffle_actions=d.get("shuffle_actions", False),
        )
    if d.get("name", "igsm") == "igsm_edit":
        kw = dict(igsm_kwargs)
        kw.pop("shuffle_actions", None)
        kw.pop("n_alt", None)
        kw.pop("geo_rank_factual_only", None)
        kw.pop("macro_alt_k", None)
        kw.pop("macro_alt_horizon", None)
        kw.pop("all_action_supervision", None)
        return EditDataset(
            vocab, size=size, seed=seed,
            max_wrong=d.max_wrong, max_missing=d.max_missing,
            max_extra=d.max_extra, vandal_prob=d.vandal_prob,
            max_vandal=d.max_vandal, n_alt=d.get("n_alt", 0), **kw,
        )
    return IGSMDataset(vocab, size=size, seed=seed, **igsm_kwargs)


def collate_for(cfg):
    """Returns the collate function matching the configured dataset."""
    return (
        collate_edits
        if cfg.data.get("name", "igsm") in {"igsm_edit", "igsm_real_token_edit"}
        else collate
    )
