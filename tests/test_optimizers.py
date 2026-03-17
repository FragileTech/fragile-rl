import math

import pytest
import torch

from fragile.core.optimizers import ThermodynamicAdam


def test_thermodynamic_adam_decreases_quadratic() -> None:
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = ThermodynamicAdam(
        [param],
        lr=0.05,
        temperature_decay=0.0,
        alignment_damping=1.0,
        trust_region=1.0,
    )

    def loss_fn() -> torch.Tensor:
        return 0.5 * (param**2).sum()

    loss_start = loss_fn().item()
    for _ in range(25):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()
    loss_end = loss_fn().item()

    assert loss_end < loss_start


def test_thermodynamic_adam_alignment_and_snr_gate() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = ThermodynamicAdam(
        [param],
        lr=0.1,
        betas=(0.9, 0.99),
        temperature_decay=0.0,
        alignment_damping=0.5,
        trust_region=0.0,
    )

    for grad_value in (1.0, -1.0):
        param.grad = torch.tensor([grad_value])
        optimizer.step()

    assert optimizer.last_alignment_dot is not None
    assert optimizer.last_alignment_dot < 0.0
    assert optimizer.last_snr is not None
    assert optimizer.last_snr < 0.05
    assert optimizer.last_lr_scale < 0.1


def test_thermodynamic_adam_varentropy_warmup_scales_lr() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = ThermodynamicAdam(
        [param],
        lr=0.1,
        betas=(0.0, 0.0),
        temperature_decay=0.5,
        varentropy_gamma=1.0,
        history_window=3,
        varentropy_min_history=3,
        alignment_damping=1.0,
        trust_region=0.0,
    )

    param.grad = torch.tensor([1.0])
    optimizer.step()
    assert optimizer.last_temperature_scale == pytest.approx(1.0)
    param.grad = torch.tensor([3.0])
    optimizer.step()
    assert optimizer.last_temperature_scale == pytest.approx(1.0)
    param.grad = torch.tensor([5.0])
    optimizer.step()

    assert optimizer.last_temperature_scale is not None
    assert optimizer.last_temperature_scale < 1.0
    assert optimizer.last_lr < optimizer.param_groups[0]["lr"]


def test_thermodynamic_adam_loss_varentropy_closure_path() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = ThermodynamicAdam(
        [param],
        lr=0.1,
        temperature_decay=0.5,
        varentropy_gamma=1.0,
        history_window=2,
        varentropy_min_history=2,
        use_loss_varentropy=True,
        alignment_damping=1.0,
        trust_region=0.0,
    )

    def loss_fn() -> torch.Tensor:
        return 0.5 * (param**2).sum()

    optimizer.zero_grad()
    loss = loss_fn()
    expected_loss = loss.item()
    loss.backward()
    optimizer.step(closure=loss_fn)
    assert optimizer.last_metric_value == pytest.approx(expected_loss)
    assert optimizer.last_temperature_scale == pytest.approx(1.0)

    optimizer.zero_grad()
    loss = loss_fn()
    expected_loss = loss.item()
    loss.backward()
    optimizer.step(closure=loss_fn)

    assert optimizer.last_metric_value == pytest.approx(expected_loss)
    assert optimizer.last_temperature_scale is not None
    assert optimizer.last_temperature_scale < 1.0


def test_thermodynamic_adam_trust_region_and_conduction() -> None:
    hot = torch.nn.Parameter(torch.tensor([1.0]))
    cold = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = ThermodynamicAdam(
        [
            {"params": [hot], "lr": 0.1},
            {"params": [cold], "lr": 1e-6},
        ],
        betas=(0.0, 0.0),
        temperature_decay=0.0,
        alignment_damping=1.0,
        trust_region=0.01,
        thermal_conductivity=1.0,
        snr_eps=1e-12,
    )

    hot.grad = torch.tensor([10.0])
    cold.grad = torch.tensor([0.0])
    optimizer.step()

    assert hot.item() == pytest.approx(0.99, rel=1e-3)

    expected_lr = math.sqrt(0.1 * 1e-6)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr, rel=1e-6)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(expected_lr, rel=1e-6)

    assert optimizer.last_temperature_scale is not None
    assert optimizer.last_temperature_scale == pytest.approx(1.0)
