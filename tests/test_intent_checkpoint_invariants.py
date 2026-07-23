import torch

from textjepa.models.ema import EMATeacher


def test_ema_teacher_remains_eval_after_parent_train():
    parent = torch.nn.Module()
    parent.target = EMATeacher(torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.Dropout(0.0),
    ))
    parent.train()
    assert not parent.target.training
    assert not parent.target.module.training
