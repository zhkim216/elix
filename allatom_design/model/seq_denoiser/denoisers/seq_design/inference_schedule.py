"""Inference-time temperature schedules for Potts sequence design."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


ScheduleName = Literal[
    "linear_T",
    "linear_beta",
    "constant_sum_kl",
    "heat_capacity",
]


@dataclass(frozen=True)
class TemperatureSchedule:
    """A scalar 1/T schedule with a terminal-temperature hold.

    ``anneal_transitions`` is the number of transitions needed to reach the
    endpoint.  For the 500-step experiment, proposal steps 0..449 use the
    pre-endpoint values and proposal steps 450..499 use the endpoint.
    """

    name: Literal["linear_T", "linear_beta"]
    start_temperature: float = 1.0
    end_temperature: float = 0.1
    total_steps: int = 500
    end_hold_steps: int = 50

    def __post_init__(self) -> None:
        if self.name not in {"linear_T", "linear_beta"}:
            raise ValueError(f"Unsupported scalar schedule: {self.name!r}")
        if not np.isfinite(self.start_temperature) or self.start_temperature <= 0:
            raise ValueError("start_temperature must be positive and finite")
        if not np.isfinite(self.end_temperature) or self.end_temperature <= 0:
            raise ValueError("end_temperature must be positive and finite")
        if self.total_steps < 2:
            raise ValueError("total_steps must be at least 2")
        if self.end_hold_steps < 1 or self.end_hold_steps >= self.total_steps:
            raise ValueError("end_hold_steps must be in [1, total_steps)")

    @property
    def anneal_transitions(self) -> int:
        return self.total_steps - self.end_hold_steps

    def betas(self) -> np.ndarray:
        start_beta = 1.0 / self.start_temperature
        end_beta = 1.0 / self.end_temperature
        if self.name == "linear_beta":
            anneal_grid = np.linspace(
                start_beta,
                end_beta,
                self.anneal_transitions + 1,
                dtype=np.float64,
            )
            anneal_values = anneal_grid[:-1]
        else:
            temperature_grid = np.linspace(
                self.start_temperature,
                self.end_temperature,
                self.anneal_transitions + 1,
                dtype=np.float64,
            )
            anneal_values = 1.0 / temperature_grid[:-1]
        values = np.concatenate(
            (
                anneal_values,
                np.full(self.end_hold_steps, end_beta, dtype=np.float64),
            )
        )
        self._validate_values(values, start_beta=start_beta, end_beta=end_beta)
        return values.astype(np.float32)

    def temperatures(self) -> np.ndarray:
        values = 1.0 / self.betas().astype(np.float64)
        return values.astype(np.float32)

    def _validate_values(
        self,
        values: np.ndarray,
        *,
        start_beta: float,
        end_beta: float,
    ) -> None:
        if values.shape != (self.total_steps,):
            raise AssertionError("temperature schedule length mismatch")
        if not np.isfinite(values).all() or bool((values <= 0).any()):
            raise AssertionError("temperature schedule contains invalid beta values")
        if not np.isclose(values[0], start_beta):
            raise AssertionError("temperature schedule does not start at start_beta")
        if not np.allclose(values[self.anneal_transitions :], end_beta):
            raise AssertionError("temperature schedule terminal hold is invalid")
        if bool((np.diff(values) < -1e-8).any()):
            raise AssertionError("beta schedule must be monotone nondecreasing")


@dataclass(frozen=True)
class InitialEnergyCalibration:
    """Initial random-sequence energies used to set per-input endpoints."""

    probe_total_energies: torch.Tensor  # [probes, B]
    probe_sequences: torch.Tensor | None  # [probes, B, N]
    lengths: torch.Tensor  # [B]
    mean_total_energy: torch.Tensor  # [B]
    mean_energy_per_residue: torch.Tensor  # [B]
    start_temperature: torch.Tensor  # [B]
    end_temperature: torch.Tensor  # [B]
    start_reduced_energy: float
    end_reduced_energy: float

    def batch_dict(self, batch_index: int) -> dict[str, object]:
        batch_size = self.probe_total_energies.shape[1]
        if batch_index < 0 or batch_index >= batch_size:
            raise IndexError(f"batch_index={batch_index} outside batch size {batch_size}")

        output: dict[str, object] = {
            "endpoint_mode": "initial_random_mean_total_U_per_residue",
            "calibration_num_sequences": int(self.probe_total_energies.shape[0]),
            "calibration_probe_total_U": self.probe_total_energies[
                :, batch_index
            ].detach().cpu().tolist(),
            "calibration_length": int(self.lengths[batch_index].detach().cpu()),
            "calibration_mean_total_U": float(
                self.mean_total_energy[batch_index].detach().cpu()
            ),
            "calibration_mean_energy_per_residue": float(
                self.mean_energy_per_residue[batch_index].detach().cpu()
            ),
            "start_temperature": float(
                self.start_temperature[batch_index].detach().cpu()
            ),
            "end_temperature": float(
                self.end_temperature[batch_index].detach().cpu()
            ),
            "start_reduced_energy": float(self.start_reduced_energy),
            "end_reduced_energy": float(self.end_reduced_energy),
        }
        if self.probe_sequences is not None:
            output["calibration_probe_sequences"] = self.probe_sequences[
                :, batch_index
            ].detach().cpu().tolist()
        return output


@dataclass(frozen=True)
class HeatCapacityCalibration:
    """T=1 endpoint local-energy curve used to build an adaptive schedule."""

    curve_temperatures: torch.Tensor  # [curve_points]
    heat_capacity: torch.Tensor  # [curve_points, B]
    effective_heat_capacity: torch.Tensor  # [curve_points, B]
    alpha: torch.Tensor  # [B]
    normalization_error: torch.Tensor  # [B]
    curve_floor_count: torch.Tensor  # [B]
    schedule_floor_count: torch.Tensor  # [B]
    initial_unique_sequences: torch.Tensor  # [B]
    annealed_unique_sequences: torch.Tensor  # [B]
    num_sequences: int
    anneal_temperature: float
    anneal_steps: int
    heat_capacity_floor: float

    def batch_dict(self, batch_index: int) -> dict[str, object]:
        batch_size = self.heat_capacity.shape[1]
        if batch_index < 0 or batch_index >= batch_size:
            raise IndexError(f"batch_index={batch_index} outside batch size {batch_size}")

        return {
            "heat_capacity_calibration_num_sequences": self.num_sequences,
            "heat_capacity_calibration_anneal_temperature": self.anneal_temperature,
            "heat_capacity_calibration_anneal_steps": self.anneal_steps,
            "heat_capacity_floor": self.heat_capacity_floor,
            "heat_capacity_alpha": float(self.alpha[batch_index].detach().cpu()),
            "heat_capacity_normalization_error": float(
                self.normalization_error[batch_index].detach().cpu()
            ),
            "heat_capacity_curve_floor_count": int(
                self.curve_floor_count[batch_index].detach().cpu()
            ),
            "heat_capacity_schedule_floor_count": int(
                self.schedule_floor_count[batch_index].detach().cpu()
            ),
            "heat_capacity_initial_unique_sequences": int(
                self.initial_unique_sequences[batch_index].detach().cpu()
            ),
            "heat_capacity_annealed_unique_sequences": int(
                self.annealed_unique_sequences[batch_index].detach().cpu()
            ),
            "heat_capacity_curve_temperature": self.curve_temperatures.detach()
            .cpu()
            .tolist(),
            "heat_capacity_curve": self.heat_capacity[:, batch_index]
            .detach()
            .cpu()
            .tolist(),
            "heat_capacity_curve_effective": self.effective_heat_capacity[
                :, batch_index
            ]
            .detach()
            .cpu()
            .tolist(),
        }


@dataclass(frozen=True)
class InferenceScheduleTrace:
    """Per-observation schedule values and frozen-energy diagnostics."""

    name: ScheduleName
    temperatures: torch.Tensor  # [steps, B]
    betas: torch.Tensor  # [steps, B]
    delta_beta: torch.Tensor  # [steps, B], transition after this proposal step
    variance_sum: torch.Tensor  # [steps, B]
    approximate_sum_kl: torch.Tensor  # [steps, B]
    exact_sum_kl: torch.Tensor  # [steps, B]
    target_sum_kl: torch.Tensor  # [B], NaN for non-adaptive schedules
    num_sites: torch.Tensor  # [B]
    variance_floor_used: torch.Tensor  # [steps, B]
    anneal_transitions: int
    end_hold_steps: int
    initial_energy_calibration: InitialEnergyCalibration | None = None
    heat_capacity_calibration: HeatCapacityCalibration | None = None

    def batch_dict(self, batch_index: int) -> dict[str, object]:
        """Return a CSV-friendly trace for one input observation."""

        batch_size = self.temperatures.shape[1]
        if batch_index < 0 or batch_index >= batch_size:
            raise IndexError(f"batch_index={batch_index} outside batch size {batch_size}")

        def _values(tensor: torch.Tensor) -> list:
            return tensor[:, batch_index].detach().cpu().tolist()

        target = float(self.target_sum_kl[batch_index].detach().cpu())
        output = {
            "schedule": self.name,
            "anneal_transitions": self.anneal_transitions,
            "end_hold_steps": self.end_hold_steps,
            "num_sites": int(self.num_sites[batch_index].detach().cpu()),
            "target_sum_kl": target,
            "temperature": _values(self.temperatures),
            "beta": _values(self.betas),
            "delta_beta": _values(self.delta_beta),
            "variance_sum": _values(self.variance_sum),
            "approximate_sum_kl": _values(self.approximate_sum_kl),
            "exact_sum_kl": _values(self.exact_sum_kl),
            "variance_floor_used": _values(self.variance_floor_used),
        }
        if self.initial_energy_calibration is not None:
            output.update(self.initial_energy_calibration.batch_dict(batch_index))
        if self.heat_capacity_calibration is not None:
            output.update(self.heat_capacity_calibration.batch_dict(batch_index))
        return output


def linear_T(
    *,
    start_temperature: float = 1.0,
    end_temperature: float = 0.1,
    total_steps: int = 500,
    end_hold_steps: int = 50,
) -> TemperatureSchedule:
    return TemperatureSchedule(
        name="linear_T",
        start_temperature=start_temperature,
        end_temperature=end_temperature,
        total_steps=total_steps,
        end_hold_steps=end_hold_steps,
    )


def linear_beta(
    *,
    start_temperature: float = 1.0,
    end_temperature: float = 0.1,
    total_steps: int = 500,
    end_hold_steps: int = 50,
) -> TemperatureSchedule:
    return TemperatureSchedule(
        name="linear_beta",
        start_temperature=start_temperature,
        end_temperature=end_temperature,
        total_steps=total_steps,
        end_hold_steps=end_hold_steps,
    )


def get_temperature_schedule(
    name: str,
    **kwargs,
) -> TemperatureSchedule:
    if name == "linear_T":
        return linear_T(**kwargs)
    if name == "linear_beta":
        return linear_beta(**kwargs)
    choices = "linear_T, linear_beta"
    raise ValueError(f"unknown scalar temperature schedule {name!r}; choose {choices}")


def _validate_energy_inputs(
    local_energy: torch.Tensor,
    legal_state_mask: torch.Tensor,
    site_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if local_energy.ndim != 3:
        raise ValueError("local_energy must have shape [B,N,Q]")
    if legal_state_mask.shape != local_energy.shape:
        raise ValueError("legal_state_mask must match local_energy shape")
    if site_mask.shape != local_energy.shape[:2]:
        raise ValueError("site_mask must have shape [B,N]")
    legal_state_mask = legal_state_mask.to(device=local_energy.device, dtype=torch.bool)
    site_mask = site_mask.to(device=local_energy.device, dtype=torch.bool)
    legal_counts = legal_state_mask.sum(dim=-1)
    if bool((legal_counts[site_mask] < 2).any()):
        raise ValueError("every scheduled site must have at least two legal states")
    if bool((site_mask.sum(dim=-1) < 1).any()):
        raise ValueError("every observation must contain at least one scheduled site")
    if not bool(torch.isfinite(local_energy[legal_state_mask]).all()):
        raise ValueError("legal local energies must be finite")
    return local_energy.double(), legal_state_mask, site_mask


def _site_statistics(
    local_energy: torch.Tensor,
    legal_state_mask: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-site mean energy, variance, and log partition function."""

    logits = -beta[:, None, None] * local_energy
    logits = logits.masked_fill(~legal_state_mask, -torch.inf)
    log_partition = torch.logsumexp(logits, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    safe_energy = torch.where(legal_state_mask, local_energy, 0.0)
    mean = (probabilities * safe_energy).sum(dim=-1)
    second_moment = (probabilities * safe_energy.square()).sum(dim=-1)
    variance = (second_moment - mean.square()).clamp_min(0.0)
    return mean, variance, log_partition


def _variance_sum(
    local_energy: torch.Tensor,
    legal_state_mask: torch.Tensor,
    site_mask: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    _, variance, _ = _site_statistics(local_energy, legal_state_mask, beta)
    # Unsampled padding can have no legal states, which makes its softmax
    # statistics NaN. Multiplication by a false mask does not clear NaN.
    return torch.where(site_mask, variance, 0.0).sum(dim=-1)


def _validate_heat_capacity_inputs(
    annealed_local_energy: torch.Tensor,
    legal_state_mask: torch.Tensor,
    site_mask: torch.Tensor,
    initial_probe_sequences: torch.Tensor,
    annealed_probe_sequences: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if annealed_local_energy.ndim != 4:
        raise ValueError("annealed_local_energy must have shape [R,B,N,Q]")
    num_sequences, batch_size, num_sites, num_states = annealed_local_energy.shape
    if num_sequences < 2:
        raise ValueError("heat-capacity calibration requires at least two sequences")
    if legal_state_mask.shape != (batch_size, num_sites, num_states):
        raise ValueError("legal_state_mask must have shape [B,N,Q]")
    if site_mask.shape != (batch_size, num_sites):
        raise ValueError("site_mask must have shape [B,N]")
    expected_sequence_shape = (num_sequences, batch_size, num_sites)
    if initial_probe_sequences.shape != expected_sequence_shape:
        raise ValueError(
            "initial_probe_sequences must have shape [R,B,N], got "
            f"{tuple(initial_probe_sequences.shape)}"
        )
    if annealed_probe_sequences.shape != expected_sequence_shape:
        raise ValueError(
            "annealed_probe_sequences must have shape [R,B,N], got "
            f"{tuple(annealed_probe_sequences.shape)}"
        )

    local_energy = annealed_local_energy.double()
    legal_state_mask = legal_state_mask.to(
        device=local_energy.device, dtype=torch.bool
    )
    site_mask = site_mask.to(device=local_energy.device, dtype=torch.bool)
    legal_counts = legal_state_mask.sum(dim=-1)
    if bool((legal_counts[site_mask] < 2).any()):
        raise ValueError("every heat-capacity site must have at least two legal states")
    if bool((site_mask.sum(dim=-1) < 1).any()):
        raise ValueError("every observation must contain a heat-capacity site")
    expanded_legal = legal_state_mask.unsqueeze(0).expand(num_sequences, -1, -1, -1)
    if not bool(torch.isfinite(local_energy[expanded_legal]).all()):
        raise ValueError("legal annealed local energies must be finite")

    initial_probe_sequences = initial_probe_sequences.to(device=local_energy.device)
    annealed_probe_sequences = annealed_probe_sequences.to(device=local_energy.device)
    initial_unique = torch.tensor(
        [
            torch.unique(initial_probe_sequences[:, batch_index], dim=0).shape[0]
            for batch_index in range(batch_size)
        ],
        device=local_energy.device,
        dtype=torch.long,
    )
    annealed_unique = torch.tensor(
        [
            torch.unique(annealed_probe_sequences[:, batch_index], dim=0).shape[0]
            for batch_index in range(batch_size)
        ],
        device=local_energy.device,
        dtype=torch.long,
    )
    if bool((initial_unique != num_sequences).any()):
        raise ValueError(
            "heat-capacity initial probes must be distinct for every observation; "
            f"unique counts={initial_unique.detach().cpu().tolist()}"
        )
    if bool((annealed_unique != num_sequences).any()):
        raise ValueError(
            "heat-capacity annealed probes must remain distinct for every observation; "
            f"unique counts={annealed_unique.detach().cpu().tolist()}"
        )
    return (
        local_energy,
        legal_state_mask,
        site_mask,
        initial_unique,
        annealed_unique,
    )


def _heat_capacity_curve(
    *,
    annealed_local_energy: torch.Tensor,
    legal_state_mask: torch.Tensor,
    site_mask: torch.Tensor,
    temperatures: torch.Tensor,
) -> torch.Tensor:
    """Return mean_R sum_i Var_q[U_i] / T^2 with shape [K,B]."""

    num_sequences, batch_size, num_sites, num_states = annealed_local_energy.shape
    flat_energy = annealed_local_energy.reshape(
        num_sequences * batch_size, num_sites, num_states
    )
    flat_legal = (
        legal_state_mask.unsqueeze(0)
        .expand(num_sequences, -1, -1, -1)
        .reshape(num_sequences * batch_size, num_sites, num_states)
    )
    flat_sites = (
        site_mask.unsqueeze(0)
        .expand(num_sequences, -1, -1)
        .reshape(num_sequences * batch_size, num_sites)
    )
    rows = []
    for temperature in temperatures:
        beta = torch.full(
            (num_sequences * batch_size,),
            1.0 / float(temperature),
            device=flat_energy.device,
            dtype=flat_energy.dtype,
        )
        variance = _variance_sum(flat_energy, flat_legal, flat_sites, beta)
        rows.append(
            variance.reshape(num_sequences, batch_size).mean(dim=0)
            / temperature.square()
        )
    heat_capacity = torch.stack(rows, dim=0)
    if not bool(torch.isfinite(heat_capacity).all()) or bool(
        (heat_capacity < 0).any()
    ):
        raise FloatingPointError("local heat-capacity curve is non-finite or negative")
    return heat_capacity


def _interpolate_temperature_curve(
    query_temperature: torch.Tensor,
    curve_temperatures: torch.Tensor,
    curve_values: torch.Tensor,
) -> torch.Tensor:
    """Linearly interpolate per-observation values on a descending T grid."""

    ascending_temperature = curve_temperatures.flip(0)
    ascending_values = curve_values.flip(0)
    query = query_temperature.clamp(
        min=float(ascending_temperature[0]),
        max=float(ascending_temperature[-1]),
    )
    upper = torch.searchsorted(ascending_temperature, query, right=False)
    upper = upper.clamp(min=1, max=len(ascending_temperature) - 1)
    lower = upper - 1
    batch_index = torch.arange(
        query.shape[0], device=query.device, dtype=torch.long
    )
    low_temperature = ascending_temperature[lower]
    high_temperature = ascending_temperature[upper]
    weight = (query - low_temperature) / (high_temperature - low_temperature)
    low_value = ascending_values[lower, batch_index]
    high_value = ascending_values[upper, batch_index]
    return low_value + weight * (high_value - low_value)


def _integrate_heat_capacity_temperature(
    *,
    alpha: torch.Tensor,
    start_temperature: float,
    end_temperature: float,
    anneal_transitions: int,
    curve_temperatures: torch.Tensor,
    effective_heat_capacity: torch.Tensor,
    return_grid: bool,
) -> torch.Tensor:
    temperature = torch.full_like(alpha, float(start_temperature))
    grid = [temperature] if return_grid else None
    total_temperature_change = float(end_temperature - start_temperature)
    for _ in range(anneal_transitions):
        capacity = _interpolate_temperature_curve(
            temperature,
            curve_temperatures,
            effective_heat_capacity,
        )
        temperature = temperature + alpha * total_temperature_change / capacity
        if return_grid:
            assert grid is not None
            grid.append(temperature)
    if return_grid:
        assert grid is not None
        return torch.stack(grid, dim=0)
    return temperature


def _solve_heat_capacity_temperature_grid(
    *,
    curve_temperatures: torch.Tensor,
    effective_heat_capacity: torch.Tensor,
    start_temperature: float,
    end_temperature: float,
    anneal_transitions: int,
    bisection_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = effective_heat_capacity.shape[1]
    low = torch.zeros(
        batch_size,
        device=effective_heat_capacity.device,
        dtype=effective_heat_capacity.dtype,
    )
    high = effective_heat_capacity.mean(dim=0) / anneal_transitions
    high = high.clamp_min(torch.finfo(high.dtype).tiny)
    for _ in range(64):
        terminal = _integrate_heat_capacity_temperature(
            alpha=high,
            start_temperature=start_temperature,
            end_temperature=end_temperature,
            anneal_transitions=anneal_transitions,
            curve_temperatures=curve_temperatures,
            effective_heat_capacity=effective_heat_capacity,
            return_grid=False,
        )
        needs_growth = terminal > end_temperature
        if not bool(needs_growth.any()):
            break
        high = torch.where(needs_growth, high * 2.0, high)
    else:
        raise RuntimeError("failed to bracket heat-capacity alpha")

    for _ in range(bisection_iterations):
        midpoint = 0.5 * (low + high)
        terminal = _integrate_heat_capacity_temperature(
            alpha=midpoint,
            start_temperature=start_temperature,
            end_temperature=end_temperature,
            anneal_transitions=anneal_transitions,
            curve_temperatures=curve_temperatures,
            effective_heat_capacity=effective_heat_capacity,
            return_grid=False,
        )
        too_warm = terminal > end_temperature
        low = torch.where(too_warm, midpoint, low)
        high = torch.where(too_warm, high, midpoint)

    alpha = 0.5 * (low + high)
    temperature_grid = _integrate_heat_capacity_temperature(
        alpha=alpha,
        start_temperature=start_temperature,
        end_temperature=end_temperature,
        anneal_transitions=anneal_transitions,
        curve_temperatures=curve_temperatures,
        effective_heat_capacity=effective_heat_capacity,
        return_grid=True,
    )
    endpoint_error = (temperature_grid[-1] - end_temperature).abs()
    if bool((endpoint_error > 1e-8).any()):
        raise RuntimeError(
            "heat-capacity endpoint solve failed: "
            f"max error={float(endpoint_error.max()):.3e}"
        )
    if bool((torch.diff(temperature_grid, dim=0) > 1e-10).any()):
        raise RuntimeError("heat-capacity schedule is not monotone nonincreasing")
    temperature_grid = temperature_grid.clone()
    temperature_grid[-1] = float(end_temperature)

    step_capacity = torch.stack(
        [
            _interpolate_temperature_curve(
                temperature_grid[step],
                curve_temperatures,
                effective_heat_capacity,
            )
            for step in range(anneal_transitions)
        ],
        dim=0,
    )
    normalization_error = (
        alpha * step_capacity.reciprocal().sum(dim=0) - 1.0
    ).abs()
    if bool((normalization_error > 1e-8).any()):
        raise RuntimeError(
            "heat-capacity alpha normalization failed: "
            f"max error={float(normalization_error.max()):.3e}"
        )
    return temperature_grid, alpha, normalization_error


def _integrate_constant_kl_beta(
    *,
    local_energy: torch.Tensor,
    legal_state_mask: torch.Tensor,
    site_mask: torch.Tensor,
    target_sum_kl: torch.Tensor,
    start_beta: float,
    end_beta: float,
    anneal_transitions: int,
    variance_floor: float,
    return_grid: bool,
) -> torch.Tensor:
    beta = torch.full_like(target_sum_kl, float(start_beta))
    beta_grid = [beta] if return_grid else None
    for _ in range(anneal_transitions):
        active = beta < end_beta
        variance_sum = _variance_sum(
            local_energy,
            legal_state_mask,
            site_mask,
            beta.clamp(max=end_beta),
        )
        delta_beta = torch.sqrt(
            2.0 * target_sum_kl / variance_sum.clamp_min(variance_floor)
        )
        beta = torch.where(active, beta + delta_beta, beta)
        if return_grid:
            assert beta_grid is not None
            beta_grid.append(beta)
    if return_grid:
        assert beta_grid is not None
        return torch.stack(beta_grid, dim=0)
    return beta


def _constant_sum_kl_beta_grid(
    *,
    local_energy: torch.Tensor,
    legal_state_mask: torch.Tensor,
    site_mask: torch.Tensor,
    start_beta: float,
    end_beta: float,
    anneal_transitions: int,
    variance_floor: float,
    bisection_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = local_energy.shape[0]
    start = torch.full(
        (batch_size,), start_beta, device=local_energy.device, dtype=local_energy.dtype
    )
    initial_variance = _variance_sum(
        local_energy, legal_state_mask, site_mask, start
    )
    if bool((initial_variance <= 0).any()):
        raise ValueError("constant_sum_kl requires positive initial summed variance")

    linear_delta = (end_beta - start_beta) / anneal_transitions
    low = torch.zeros_like(initial_variance)
    high = 0.5 * linear_delta * linear_delta * initial_variance
    high = high.clamp_min(torch.finfo(high.dtype).tiny)
    for _ in range(64):
        terminal = _integrate_constant_kl_beta(
            local_energy=local_energy,
            legal_state_mask=legal_state_mask,
            site_mask=site_mask,
            target_sum_kl=high,
            start_beta=start_beta,
            end_beta=end_beta,
            anneal_transitions=anneal_transitions,
            variance_floor=variance_floor,
            return_grid=False,
        )
        needs_growth = terminal < end_beta
        if not bool(needs_growth.any()):
            break
        high = torch.where(needs_growth, high * 2.0, high)
    else:
        raise RuntimeError("failed to bracket constant_sum_kl target")

    for _ in range(bisection_iterations):
        midpoint = 0.5 * (low + high)
        terminal = _integrate_constant_kl_beta(
            local_energy=local_energy,
            legal_state_mask=legal_state_mask,
            site_mask=site_mask,
            target_sum_kl=midpoint,
            start_beta=start_beta,
            end_beta=end_beta,
            anneal_transitions=anneal_transitions,
            variance_floor=variance_floor,
            return_grid=False,
        )
        below = terminal < end_beta
        low = torch.where(below, midpoint, low)
        high = torch.where(below, high, midpoint)

    target = 0.5 * (low + high)
    beta_grid = _integrate_constant_kl_beta(
        local_energy=local_energy,
        legal_state_mask=legal_state_mask,
        site_mask=site_mask,
        target_sum_kl=target,
        start_beta=start_beta,
        end_beta=end_beta,
        anneal_transitions=anneal_transitions,
        variance_floor=variance_floor,
        return_grid=True,
    )
    endpoint_error = (beta_grid[-1] - end_beta).abs()
    if bool((endpoint_error > 1e-6).any()):
        raise RuntimeError(
            "constant_sum_kl endpoint solve failed: "
            f"max error={float(endpoint_error.max()):.3e}"
        )
    beta_grid = beta_grid.clone()
    beta_grid[-1] = float(end_beta)
    return beta_grid, target


def _diagnose_schedule(
    *,
    name: ScheduleName,
    betas: torch.Tensor,
    local_energy: torch.Tensor,
    legal_state_mask: torch.Tensor,
    site_mask: torch.Tensor,
    target_sum_kl: torch.Tensor,
    anneal_transitions: int,
    end_hold_steps: int,
    variance_floor: float,
    initial_energy_calibration: InitialEnergyCalibration | None = None,
) -> InferenceScheduleTrace:
    steps, batch_size = betas.shape
    delta_beta = torch.zeros_like(betas)
    delta_beta[:-1] = betas[1:] - betas[:-1]
    variance_rows = []
    exact_rows = []
    floor_rows = []
    for step in range(steps):
        beta = betas[step]
        mean, variance, log_partition = _site_statistics(
            local_energy, legal_state_mask, beta
        )
        variance_sum = torch.where(site_mask, variance, 0.0).sum(dim=-1)
        variance_rows.append(variance_sum)
        floor_rows.append(variance_sum < variance_floor)
        if step + 1 < steps:
            next_beta = betas[step + 1]
            _, _, next_log_partition = _site_statistics(
                local_energy, legal_state_mask, next_beta
            )
            site_kl = (
                (next_beta - beta)[:, None] * mean
                + next_log_partition
                - log_partition
            )
            exact_rows.append(
                torch.where(site_mask, site_kl, 0.0)
                .sum(dim=-1)
                .clamp_min(0.0)
            )
        else:
            exact_rows.append(torch.zeros(batch_size, device=betas.device, dtype=betas.dtype))
    variance_sum = torch.stack(variance_rows, dim=0)
    exact_sum_kl = torch.stack(exact_rows, dim=0)
    approximate_sum_kl = (
        0.5 * delta_beta.square() * variance_sum.clamp_min(variance_floor)
    )
    return InferenceScheduleTrace(
        name=name,
        temperatures=(1.0 / betas).float(),
        betas=betas.float(),
        delta_beta=delta_beta.float(),
        variance_sum=variance_sum.float(),
        approximate_sum_kl=approximate_sum_kl.float(),
        exact_sum_kl=exact_sum_kl.float(),
        target_sum_kl=target_sum_kl.float(),
        num_sites=site_mask.sum(dim=-1),
        variance_floor_used=torch.stack(floor_rows, dim=0),
        anneal_transitions=anneal_transitions,
        end_hold_steps=end_hold_steps,
        initial_energy_calibration=initial_energy_calibration,
    )


def build_inference_schedule_trace(
    name: ScheduleName,
    *,
    local_energy: torch.Tensor,
    legal_state_mask: torch.Tensor,
    site_mask: torch.Tensor,
    start_temperature: float = 1.0,
    end_temperature: float = 0.1,
    total_steps: int = 500,
    end_hold_steps: int = 50,
    variance_floor: float = 1e-12,
    bisection_iterations: int = 48,
) -> InferenceScheduleTrace:
    """Build one schedule per observation from frozen initial local energies."""

    local_energy, legal_state_mask, site_mask = _validate_energy_inputs(
        local_energy, legal_state_mask, site_mask
    )
    if not np.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("variance_floor must be positive and finite")
    if bisection_iterations < 1:
        raise ValueError("bisection_iterations must be positive")
    anneal_transitions = total_steps - end_hold_steps
    if anneal_transitions < 1:
        raise ValueError("total_steps must exceed end_hold_steps")
    start_beta = 1.0 / float(start_temperature)
    end_beta = 1.0 / float(end_temperature)
    if end_beta <= start_beta:
        raise ValueError("end beta must be greater than start beta")
    batch_size = local_energy.shape[0]

    if name in {"linear_T", "linear_beta"}:
        scalar = get_temperature_schedule(
            name,
            start_temperature=start_temperature,
            end_temperature=end_temperature,
            total_steps=total_steps,
            end_hold_steps=end_hold_steps,
        )
        beta_values = torch.as_tensor(
            scalar.betas(), device=local_energy.device, dtype=local_energy.dtype
        )
        betas = beta_values[:, None].expand(-1, batch_size).clone()
        target = torch.full(
            (batch_size,), torch.nan, device=local_energy.device, dtype=local_energy.dtype
        )
    elif name == "constant_sum_kl":
        beta_grid, target = _constant_sum_kl_beta_grid(
            local_energy=local_energy,
            legal_state_mask=legal_state_mask,
            site_mask=site_mask,
            start_beta=start_beta,
            end_beta=end_beta,
            anneal_transitions=anneal_transitions,
            variance_floor=variance_floor,
            bisection_iterations=bisection_iterations,
        )
        betas = torch.cat(
            (
                beta_grid[:-1],
                torch.full(
                    (end_hold_steps, batch_size),
                    end_beta,
                    device=local_energy.device,
                    dtype=local_energy.dtype,
                ),
            ),
            dim=0,
        )
    else:
        choices = "linear_T, linear_beta, constant_sum_kl"
        raise ValueError(f"unknown inference schedule {name!r}; choose {choices}")

    return _diagnose_schedule(
        name=name,
        betas=betas,
        local_energy=local_energy,
        legal_state_mask=legal_state_mask,
        site_mask=site_mask,
        target_sum_kl=target,
        anneal_transitions=anneal_transitions,
        end_hold_steps=end_hold_steps,
        variance_floor=variance_floor,
    )


def build_heat_capacity_schedule_trace(
    *,
    annealed_local_energy: torch.Tensor,
    legal_state_mask: torch.Tensor,
    site_mask: torch.Tensor,
    initial_probe_sequences: torch.Tensor,
    annealed_probe_sequences: torch.Tensor,
    start_temperature: float = 3.0,
    end_temperature: float = 0.01,
    total_steps: int = 500,
    end_hold_steps: int = 50,
    curve_points: int = 200,
    heat_capacity_floor: float = 0.1,
    bisection_iterations: int = 64,
    calibration_anneal_temperature: float = 1.0,
    calibration_anneal_steps: int = 50,
) -> InferenceScheduleTrace:
    """Build a per-observation schedule from T=1 endpoint local energies.

    The first ``total_steps - end_hold_steps`` transitions obey the discrete
    recurrence

    ``T[i+1] = T[i] + alpha * (T_end - T_start) / max(Cv(T[i]), floor)``.

    ``alpha`` is solved independently for each observation so the final
    cooling transition reaches ``end_temperature``. The final proposal steps
    then hold that endpoint temperature.
    """

    if not np.isfinite(start_temperature) or start_temperature <= 0:
        raise ValueError("start_temperature must be positive and finite")
    if not np.isfinite(end_temperature) or end_temperature <= 0:
        raise ValueError("end_temperature must be positive and finite")
    if end_temperature >= start_temperature:
        raise ValueError("end_temperature must be smaller than start_temperature")
    if total_steps < 2 or end_hold_steps < 1 or end_hold_steps >= total_steps:
        raise ValueError("total_steps must exceed end_hold_steps >= 1")
    if curve_points < 2:
        raise ValueError("curve_points must be at least 2")
    if not np.isfinite(heat_capacity_floor) or heat_capacity_floor <= 0:
        raise ValueError("heat_capacity_floor must be positive and finite")
    if bisection_iterations < 1:
        raise ValueError("bisection_iterations must be positive")
    if (
        not np.isfinite(calibration_anneal_temperature)
        or calibration_anneal_temperature <= 0
    ):
        raise ValueError("calibration_anneal_temperature must be positive and finite")
    if calibration_anneal_steps < 1:
        raise ValueError("calibration_anneal_steps must be positive")

    (
        local_energy,
        legal_state_mask,
        site_mask,
        initial_unique,
        annealed_unique,
    ) = _validate_heat_capacity_inputs(
        annealed_local_energy,
        legal_state_mask,
        site_mask,
        initial_probe_sequences,
        annealed_probe_sequences,
    )
    device = local_energy.device
    dtype = local_energy.dtype
    batch_size = local_energy.shape[1]
    num_sequences = local_energy.shape[0]
    anneal_transitions = total_steps - end_hold_steps
    curve_temperatures = torch.linspace(
        start_temperature,
        end_temperature,
        curve_points,
        device=device,
        dtype=dtype,
    )
    heat_capacity = _heat_capacity_curve(
        annealed_local_energy=local_energy,
        legal_state_mask=legal_state_mask,
        site_mask=site_mask,
        temperatures=curve_temperatures,
    )
    effective_heat_capacity = heat_capacity.clamp_min(heat_capacity_floor)
    temperature_grid, alpha, normalization_error = (
        _solve_heat_capacity_temperature_grid(
            curve_temperatures=curve_temperatures,
            effective_heat_capacity=effective_heat_capacity,
            start_temperature=start_temperature,
            end_temperature=end_temperature,
            anneal_transitions=anneal_transitions,
            bisection_iterations=bisection_iterations,
        )
    )
    temperatures = torch.cat(
        (
            temperature_grid[:-1],
            torch.full(
                (end_hold_steps, batch_size),
                end_temperature,
                device=device,
                dtype=dtype,
            ),
        ),
        dim=0,
    )
    if temperatures.shape != (total_steps, batch_size):
        raise AssertionError("heat-capacity temperature schedule length mismatch")
    if not bool(torch.isfinite(temperatures).all()) or bool(
        (temperatures <= 0).any()
    ):
        raise FloatingPointError("heat-capacity schedule is non-finite or nonpositive")

    scheduled_heat_capacity = torch.stack(
        [
            _interpolate_temperature_curve(
                temperatures[step], curve_temperatures, heat_capacity
            )
            for step in range(total_steps)
        ],
        dim=0,
    )
    variance_sum = scheduled_heat_capacity * temperatures.square()
    betas = temperatures.reciprocal()
    delta_beta = torch.zeros_like(betas)
    delta_beta[:-1] = betas[1:] - betas[:-1]
    approximate_sum_kl = 0.5 * delta_beta.square() * variance_sum
    exact_sum_kl = torch.full_like(temperatures, torch.nan)
    schedule_floor_used = scheduled_heat_capacity < heat_capacity_floor
    target_sum_kl = torch.full(
        (batch_size,), torch.nan, device=device, dtype=dtype
    )
    calibration = HeatCapacityCalibration(
        curve_temperatures=curve_temperatures.float(),
        heat_capacity=heat_capacity.float(),
        effective_heat_capacity=effective_heat_capacity.float(),
        alpha=alpha.float(),
        normalization_error=normalization_error.float(),
        curve_floor_count=(heat_capacity < heat_capacity_floor).sum(dim=0),
        schedule_floor_count=schedule_floor_used.sum(dim=0),
        initial_unique_sequences=initial_unique,
        annealed_unique_sequences=annealed_unique,
        num_sequences=num_sequences,
        anneal_temperature=float(calibration_anneal_temperature),
        anneal_steps=int(calibration_anneal_steps),
        heat_capacity_floor=float(heat_capacity_floor),
    )
    return InferenceScheduleTrace(
        name="heat_capacity",
        temperatures=temperatures.float(),
        betas=betas.float(),
        delta_beta=delta_beta.float(),
        variance_sum=variance_sum.float(),
        approximate_sum_kl=approximate_sum_kl.float(),
        exact_sum_kl=exact_sum_kl.float(),
        target_sum_kl=target_sum_kl.float(),
        num_sites=site_mask.sum(dim=-1),
        variance_floor_used=schedule_floor_used,
        anneal_transitions=anneal_transitions,
        end_hold_steps=end_hold_steps,
        heat_capacity_calibration=calibration,
    )


def build_energy_density_schedule_trace(
    name: Literal["linear_T", "linear_beta"],
    *,
    probe_total_energies: torch.Tensor,
    probe_sequences: torch.Tensor | None,
    lengths: torch.Tensor,
    local_energy: torch.Tensor,
    legal_state_mask: torch.Tensor,
    site_mask: torch.Tensor,
    start_reduced_energy: float = 0.01,
    end_reduced_energy: float = 10.0,
    total_steps: int = 500,
    end_hold_steps: int = 50,
    variance_floor: float = 1e-12,
) -> InferenceScheduleTrace:
    """Build per-input linear schedules from initial raw Potts energy density.

    The positive scale is ``mean(U) / L``.  Nonpositive mean initial energies
    fail explicitly rather than being silently absolutized or clipped.
    """

    if name not in {"linear_T", "linear_beta"}:
        raise ValueError(
            "energy-density endpoints require schedule 'linear_T' or 'linear_beta'"
        )
    if not np.isfinite(start_reduced_energy) or start_reduced_energy <= 0:
        raise ValueError("start_reduced_energy must be positive and finite")
    if not np.isfinite(end_reduced_energy) or end_reduced_energy <= start_reduced_energy:
        raise ValueError("end_reduced_energy must be finite and greater than start")
    if total_steps < 2 or end_hold_steps < 1 or end_hold_steps >= total_steps:
        raise ValueError("total_steps must exceed end_hold_steps >= 1")
    if not np.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("variance_floor must be positive and finite")

    local_energy, legal_state_mask, site_mask = _validate_energy_inputs(
        local_energy, legal_state_mask, site_mask
    )
    device = local_energy.device
    dtype = local_energy.dtype
    batch_size = local_energy.shape[0]
    probe_total_energies = torch.as_tensor(
        probe_total_energies, device=device, dtype=dtype
    )
    if probe_total_energies.ndim != 2 or probe_total_energies.shape[1] != batch_size:
        raise ValueError("probe_total_energies must have shape [probes,B]")
    if probe_total_energies.shape[0] < 1:
        raise ValueError("at least one probe sequence is required")
    if not bool(torch.isfinite(probe_total_energies).all()):
        raise ValueError("probe_total_energies must be finite")

    lengths = torch.as_tensor(lengths, device=device, dtype=dtype)
    if lengths.shape != (batch_size,):
        raise ValueError("lengths must have shape [B]")
    if not bool(torch.isfinite(lengths).all()) or bool((lengths <= 0).any()):
        raise ValueError("lengths must be positive and finite")

    if probe_sequences is not None:
        probe_sequences = torch.as_tensor(probe_sequences, device=device)
        expected_prefix = (
            probe_total_energies.shape[0],
            batch_size,
            local_energy.shape[1],
        )
        if probe_sequences.shape != expected_prefix:
            raise ValueError(
                "probe_sequences must have shape [probes,B,N], got "
                f"{tuple(probe_sequences.shape)}"
            )

    mean_total_energy = probe_total_energies.mean(dim=0)
    if bool((mean_total_energy <= 0).any()):
        bad = torch.nonzero(mean_total_energy <= 0, as_tuple=False).flatten().tolist()
        raise ValueError(
            "energy-density endpoints require positive mean initial Potts energy; "
            f"nonpositive batch indices={bad}; "
            f"mean energies={mean_total_energy.detach().cpu().tolist()}"
        )
    mean_energy_per_residue = mean_total_energy / lengths
    start_temperature = mean_energy_per_residue / float(start_reduced_energy)
    end_temperature = mean_energy_per_residue / float(end_reduced_energy)
    if not bool(torch.isfinite(start_temperature).all()) or not bool(
        torch.isfinite(end_temperature).all()
    ):
        raise ValueError("calibrated temperatures must be finite")

    anneal_transitions = total_steps - end_hold_steps
    fraction = torch.linspace(
        0.0,
        1.0,
        anneal_transitions + 1,
        device=device,
        dtype=dtype,
    )[:, None]
    start_beta = 1.0 / start_temperature
    end_beta = 1.0 / end_temperature
    if name == "linear_T":
        anneal_temperature = start_temperature[None] + fraction * (
            end_temperature - start_temperature
        )[None]
        anneal_beta = 1.0 / anneal_temperature
    else:
        anneal_beta = start_beta[None] + fraction * (end_beta - start_beta)[None]
    betas = torch.cat(
        (
            anneal_beta[:-1],
            end_beta[None].expand(end_hold_steps, -1),
        ),
        dim=0,
    )

    calibration = InitialEnergyCalibration(
        probe_total_energies=probe_total_energies.float(),
        probe_sequences=probe_sequences,
        lengths=lengths.float(),
        mean_total_energy=mean_total_energy.float(),
        mean_energy_per_residue=mean_energy_per_residue.float(),
        start_temperature=start_temperature.float(),
        end_temperature=end_temperature.float(),
        start_reduced_energy=float(start_reduced_energy),
        end_reduced_energy=float(end_reduced_energy),
    )
    target = torch.full((batch_size,), torch.nan, device=device, dtype=dtype)
    return _diagnose_schedule(
        name=name,
        betas=betas,
        local_energy=local_energy,
        legal_state_mask=legal_state_mask,
        site_mask=site_mask,
        target_sum_kl=target,
        anneal_transitions=anneal_transitions,
        end_hold_steps=end_hold_steps,
        variance_floor=variance_floor,
        initial_energy_calibration=calibration,
    )
