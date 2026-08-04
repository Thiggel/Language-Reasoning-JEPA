"""Shared runtime helpers for the pinned hierarchical-language pilot.

Executable scripts import these contracts from the installed ``textjepa``
package. Importing one file from another under ``scripts/`` is unreliable when
the controller runs an immutable source snapshot by path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from textjepa.analysis.compute import parameter_count
from textjepa.data.language_planning import (
    MODEL_ID,
    MODEL_REVISION,
    PAD_TOKEN_ID,
    TRANSFORMERS_VERSION,
)
from textjepa.models.hierarchical_language_jepa import (
    HIERARCHICAL_LANGUAGE_ARCHITECTURE,
    HierarchicalLanguageJEPA,
    HierarchicalLanguageJEPAConfig,
)
from textjepa.training.hierarchical_language import (
    HierarchicalLanguageLearner,
    ResearchStage,
)


def load_reference_model(device: str, dtype_name: str):
    """Load and validate the exact frozen Qwen pilot model."""
    try:
        import transformers
        from transformers import AutoModelForMultimodalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "install textjepa[language-planning] to use frozen-LM features"
        ) from error
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Transformers {TRANSFORMERS_VERSION} is required, got "
            f"{transformers.__version__}"
        )
    dtype = getattr(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_fast=True
    )
    if tokenizer.add_bos_token:
        raise RuntimeError("pinned Qwen tokenizer unexpectedly adds BOS")
    if tokenizer.pad_token_id != PAD_TOKEN_ID:
        raise RuntimeError("pinned Qwen padding ID changed")
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=dtype
    ).to(device).eval()
    return tokenizer, model


def text_parameter_count(model) -> int:
    """Count unique parameters exercised by a text-only Qwen forward."""
    if not hasattr(model, "model") or not hasattr(
        model.model, "language_model"
    ):
        return parameter_count(model)
    modules = [model.model.language_model, model.lm_head]
    seen: set[int] = set()
    total = 0
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                total += parameter.numel()
    return total


def backend_metadata() -> dict[str, object]:
    """Return weights-only-serializable execution-backend metadata."""
    flash = importlib.util.find_spec("fla") is not None
    causal_conv = importlib.util.find_spec("causal_conv1d") is not None
    return {
        "text_backend": (
            "flash_linear_attention" if flash and causal_conv
            else "torch_fallback"
        ),
        "flash_linear_attention_available": flash,
        "causal_conv1d_available": causal_conv,
        "torch_version": str(torch.__version__),
        "cuda_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
    }


def load_hierarchical_checkpoint(path: Path, device: str):
    """Load a strict-nested planning checkpoint and learned geometry."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != HIERARCHICAL_LANGUAGE_ARCHITECTURE:
        raise ValueError(
            "checkpoint does not use the strict nested E0 -> E0_to_1 tower"
        )
    config = HierarchicalLanguageJEPAConfig(**checkpoint["config"])
    model = HierarchicalLanguageJEPA(config).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    stage = ResearchStage[checkpoint["stage"]]
    experiment = checkpoint.get("experiment_config") or {}
    loss = experiment.get("loss", {})
    research = experiment.get("research", {})
    learner = HierarchicalLanguageLearner(
        model,
        stage,
        dynamics_geometry=loss.get("dynamics_geometry", "mahalanobis"),
        normalize_dynamics=bool(loss.get("normalize_dynamics", False)),
        anti_collapse=loss.get("anti_collapse", "vicreg"),
        sigreg_slices=int(loss.get("sigreg_slices", 256)),
        joint_token_sentence=bool(
            research.get("joint_token_sentence", False)
        ),
    ).to(device).eval()
    learner.load_state_dict(checkpoint["learner"])
    return model, learner
