from __future__ import annotations

from collections import deque
import math
from typing import Iterable

import torch
from torch.optim import Optimizer


class ThermodynamicAdam(Optimizer):
    """Adam-style optimizer with a thermodynamic governor.

    Implements the optimizer controls derived in
    docs/source/1_agent/03_architecture/03_optmization.md:
    - Trust-region (Mach limit) scaling.
    - Alignment-triggered step damping.
    - Varentropy brake cooling.
    - SNR gating.
    - Optional log-LR thermal conduction across param groups.

    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        temperature_decay: float = 0.003,
        temperature_floor: float = 0.1,
        varentropy_gamma: float = 1.0,
        alignment_damping: float = 0.8,
        trust_region: float = 0.05,
        trust_region_eps: float = 0.0,
        snr_eps: float = 1e-8,
        snr_floor: float = 0.05,
        thermal_conductivity: float = 0.0,
        history_window: int = 20,
        varentropy_min_history: int = 5,
        varentropy_eps: float = 1e-8,
        use_loss_varentropy: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if len(betas) != 2 or not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0.0 <= temperature_decay < 1.0:
            raise ValueError(f"Invalid temperature_decay: {temperature_decay}")
        if not 0.0 <= temperature_floor <= 1.0:
            raise ValueError(f"Invalid temperature_floor: {temperature_floor}")
        if trust_region < 0.0:
            raise ValueError(f"Invalid trust_region: {trust_region}")
        if trust_region_eps < 0.0:
            raise ValueError(f"Invalid trust_region_eps: {trust_region_eps}")
        if snr_eps < 0.0:
            raise ValueError(f"Invalid snr_eps: {snr_eps}")
        if not 0.0 <= snr_floor <= 1.0:
            raise ValueError(f"Invalid snr_floor: {snr_floor}")
        if not 0.0 <= thermal_conductivity <= 1.0:
            raise ValueError(f"Invalid thermal_conductivity: {thermal_conductivity}")
        if history_window < 1:
            raise ValueError(f"Invalid history_window: {history_window}")
        if varentropy_min_history < 1:
            raise ValueError(f"Invalid varentropy_min_history: {varentropy_min_history}")
        if varentropy_min_history > history_window:
            msg = "varentropy_min_history cannot exceed history_window"
            raise ValueError(msg)
        if varentropy_eps <= 0.0:
            raise ValueError(f"Invalid varentropy_eps: {varentropy_eps}")

        if not 0.0 < alignment_damping <= 1.0:
            raise ValueError(f"Invalid alignment_damping: {alignment_damping}")
        if varentropy_gamma < 0.0:
            raise ValueError(f"Invalid varentropy_gamma: {varentropy_gamma}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "temperature_decay": temperature_decay,
            "temperature_floor": temperature_floor,
            "varentropy_gamma": varentropy_gamma,
            "alignment_damping": alignment_damping,
            "trust_region": trust_region,
            "trust_region_eps": trust_region_eps,
            "snr_eps": snr_eps,
            "snr_floor": snr_floor,
            "thermal_conductivity": thermal_conductivity,
            "varentropy_eps": varentropy_eps,
        }
        super().__init__(params, defaults)

        self._stat_history: deque[float] = deque(maxlen=history_window)
        self._varentropy_min_history = varentropy_min_history
        self._use_loss_varentropy = use_loss_varentropy
        self.last_lr_scale: float = 1.0
        self.last_lr: float = lr
        self.last_lrs: list[float] = []
        self.last_base_lr: float = lr
        self.last_base_lrs: list[float] = []
        self.last_metric_value: float | None = None
        self.last_varentropy: float | None = None
        self.last_alignment_dot: float | None = None
        self.last_snr: float | None = None
        self.last_snrs: list[float | None] = []
        self.last_temperature_scale: float | None = None
        self._temperature_scale: float = 1.0
        self._global_step: int = 0

    @torch.no_grad()
    def step(self, closure=None, *, loss=None):  # type: ignore[override]
        """Perform a single optimization step.

        Args:
            closure: Optional callable returning the loss (used when use_loss_varentropy=True).
            loss: Optional precomputed loss value (used when use_loss_varentropy=True).
        """
        loss_value = None
        loss_out = loss
        if loss is not None:
            if self._use_loss_varentropy:
                if torch.is_tensor(loss):
                    loss_value = float(loss.detach().item())
                else:
                    loss_value = float(loss)
        elif closure is not None:
            with torch.enable_grad():
                loss_tensor = closure()
            if torch.is_tensor(loss_tensor):
                if self._use_loss_varentropy:
                    loss_value = float(loss_tensor.detach().item())
            elif self._use_loss_varentropy:
                loss_value = float(loss_tensor)
            loss_out = loss_tensor

        metric_value = None
        if self._use_loss_varentropy and loss_value is not None:
            metric_value = loss_value
        else:
            metric_value = self._compute_grad_rms()
        self.last_metric_value = metric_value

        temperature_step = 1.0
        if metric_value is not None:
            self._stat_history.append(metric_value)
            if len(self._stat_history) >= self._varentropy_min_history:
                mean = math.fsum(self._stat_history) / len(self._stat_history)
                variance = math.fsum((val - mean) ** 2 for val in self._stat_history) / len(
                    self._stat_history
                )
                varentropy = variance / (mean * mean + self.param_groups[0]["varentropy_eps"])
                self.last_varentropy = varentropy
                gamma = self.param_groups[0]["varentropy_gamma"]
                decay = self.param_groups[0]["temperature_decay"]
                if decay > 0.0:
                    temperature_step = 1.0 - decay / (1.0 + gamma * varentropy)
                    temperature_step = min(1.0, max(0.0, temperature_step))
            else:
                self.last_varentropy = None
        else:
            self.last_varentropy = None

        self._temperature_scale *= temperature_step
        if self.param_groups:
            temperature_floor = float(self.param_groups[0].get("temperature_floor", 0.0))
            self._temperature_scale = max(self._temperature_scale, temperature_floor)
        self.last_temperature_scale = self._temperature_scale

        self.last_lr_scale = 1.0
        self.last_lrs = []
        self.last_base_lrs = []
        self.last_alignment_dot = None
        self.last_snr = None
        self.last_snrs = [None] * len(self.param_groups)

        for group_index, group in enumerate(self.param_groups):
            base_lr = group["lr"]
            temperature_lr = base_lr * self._temperature_scale
            beta1, beta2 = group["betas"]

            alignment_sum = 0.0
            signal = 0.0
            energy = 0.0
            has_grad = False

            # Pass 1: alignment + update moments + collect SNR stats.
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    msg = "ThermodynamicAdam does not support sparse gradients."
                    raise RuntimeError(msg)

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                alignment_sum += torch.sum(grad * exp_avg).item()

                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                signal += float(exp_avg.pow(2).sum().item())
                energy += float(exp_avg_sq.sum().item())
                has_grad = True

            if not has_grad:
                self.last_lrs.append(base_lr)
                self.last_base_lrs.append(base_lr)
                continue

            noise = max(energy - signal, 0.0)
            snr = signal / (noise + group["snr_eps"]) if signal > 0.0 else 0.0
            snr_factor = snr / (1.0 + snr) if snr > 0.0 else 0.0
            snr_factor = max(snr_factor, group.get("snr_floor", 0.0))
            alignment_factor = group["alignment_damping"] if alignment_sum < 0.0 else 1.0
            effective_lr = temperature_lr * alignment_factor * snr_factor

            self.last_snrs[group_index] = snr
            if group_index == 0:
                self.last_alignment_dot = alignment_sum
                self.last_snr = snr
                self.last_lr_scale = effective_lr / base_lr if base_lr > 0.0 else 0.0

            # Pass 2: apply updates with trust-region scaling.
            for param in group["params"]:
                if param.grad is None:
                    continue

                if group["weight_decay"] != 0.0:
                    param.add_(param, alpha=-effective_lr * group["weight_decay"])

                state = self.state[param]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = effective_lr * math.sqrt(bias_correction2) / bias_correction1

                update = exp_avg / denom

                trust_region = group["trust_region"]
                if trust_region > 0.0 and step_size > 0.0:
                    p_norm = param.norm()
                    u_norm = update.norm() * step_size
                    max_step = trust_region * (p_norm + group["trust_region_eps"])
                    if u_norm > 0.0:
                        scale = min(1.0, max_step / (u_norm + 1e-12))
                        step_size *= scale

                param.add_(update, alpha=-step_size)

            self.last_lrs.append(effective_lr)
            self.last_base_lrs.append(base_lr)

        if self.param_groups:
            k = float(self.param_groups[0]["thermal_conductivity"])
            if k > 0.0 and len(self.param_groups) > 1:
                lrs = [g["lr"] for g in self.param_groups]
                new_lrs = list(lrs)
                for i in range(len(lrs)):
                    left = lrs[i - 1] if i > 0 else lrs[i]
                    right = lrs[i + 1] if i < len(lrs) - 1 else lrs[i]
                    log_self = math.log(lrs[i] + 1e-12)
                    log_left = math.log(left + 1e-12)
                    log_right = math.log(right + 1e-12)
                    flux = (log_left - log_self) + (log_right - log_self)
                    new_lrs[i] = math.exp(log_self + 0.5 * k * flux)
                for i, group in enumerate(self.param_groups):
                    group["lr"] = new_lrs[i]

        if self.last_lrs:
            self.last_lr = self.last_lrs[0]
        if self.last_base_lrs:
            self.last_base_lr = self.last_base_lrs[0]
        self._global_step += 1

        return loss_out

    def state_dict(self):  # type: ignore[override]
        state = super().state_dict()
        state["thermo_global_step"] = self._global_step
        state["thermo_temperature_scale"] = self._temperature_scale
        return state

    def load_state_dict(self, state_dict):  # type: ignore[override]
        self._global_step = int(state_dict.pop("thermo_global_step", 0))
        self._temperature_scale = float(state_dict.pop("thermo_temperature_scale", 1.0))
        super().load_state_dict(state_dict)

    def _compute_grad_rms(self) -> float | None:
        grad_sq_sum = 0.0
        grad_count = 0
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    msg = "ThermodynamicAdam does not support sparse gradients."
                    raise RuntimeError(msg)
                grad_sq_sum += float(grad.detach().float().pow(2).sum().item())
                grad_count += grad.numel()
        if grad_count == 0:
            return None
        return math.sqrt(grad_sq_sum / grad_count)
