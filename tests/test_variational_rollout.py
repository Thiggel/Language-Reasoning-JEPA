import torch

from scripts.audit_variational_rollout import calibration_metrics


def test_calibration_metrics_applies_frozen_spread_temperature():
    z = torch.tensor([[-2.0, 2.0], [-1.0, 1.0]])
    metrics = calibration_metrics(z, temperature=2.0)

    assert metrics["z2_raw"] == 2.5
    assert metrics["z2_calibrated"] == 0.625
    assert metrics["coverage_1sigma_raw"] == 0.5
    assert metrics["coverage_1sigma_calibrated"] == 1.0
    assert metrics["coverage_2sigma_raw"] == 1.0
    assert metrics["coverage_2sigma_calibrated"] == 1.0
