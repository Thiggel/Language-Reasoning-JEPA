from pathlib import Path

import torch
from omegaconf import OmegaConf

from textjepa.training.trainer import Trainer


class _Model(torch.nn.Linear):
    def update_teachers(self, momentum):
        del momentum


def _cfg():
    return OmegaConf.create({
        "device": "cpu",
        "train": {
            "epochs": 1, "batch_size": 1, "microbatch_size": 1,
            "lr": 1e-3, "weight_decay": 0.0, "grad_clip": 1.0,
            "warmup_steps": 0, "lr_floor": 0.0, "ema_start": 0.99,
            "ema_end": 0.999, "log_every": 1, "eval_batches": 1,
            "precision": "fp32", "checkpoint_every_steps": 1,
        },
    })


def test_optimizer_boundary_checkpoint_is_resumable(tmp_path: Path):
    model = _Model(2, 2)
    trainer = Trainer(_cfg(), model, torch.nn.Identity(), [{}], [{}], tmp_path)
    trainer.step = 7
    trainer.epoch = 0
    trainer.micro_step_in_epoch = 7
    trainer._checkpoint("last.pt", 0, {"loss": 1.0})

    payload = torch.load(tmp_path / "last.pt", weights_only=False)
    assert payload["step"] == 7
    assert payload["micro_step_in_epoch"] == 7
    assert "optimizer" in payload

    restored = Trainer(
        _cfg(), _Model(2, 2), torch.nn.Identity(), [{}], [{}], tmp_path / "new"
    )
    restored.resume(tmp_path / "last.pt")
    assert (restored.step, restored.epoch, restored.micro_step_in_epoch) == (7, 0, 7)
