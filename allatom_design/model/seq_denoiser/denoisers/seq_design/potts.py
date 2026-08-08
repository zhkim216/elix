# Copyright Generate Biomedicines, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layers for building Potts models.

This module contains layers for parameterizing Potts models from
graph embeddings.

Adapted from Chroma by Richard Shuai.
Modified for Elix by Jinho Kim.
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType
from tqdm.auto import tqdm

from allatom_design.model.seq_denoiser.denoisers.seq_design import \
    graph_utils as graph
from allatom_design.model.seq_denoiser.denoisers.seq_design.multi_head_potts import (
    SharedEdgeBottleneckMultiHeadFactorPotts,
)
from allatom_design.model.seq_denoiser.denoisers.seq_design.inference_schedule import (
    InferenceScheduleTrace,
    build_inference_schedule_trace,
)
from allatom_design.model.seq_denoiser.denoisers.seq_design.potts_proposal_dispatch import (
    PottsProposalMode,
    resolve_potts_proposal_mode,
)


PottsMixingScheme = Literal["probability", "energy", "mean_energy", "pcebm"]
POTTS_MIXING_SCHEMES = frozenset(
    {"probability", "energy", "mean_energy", "pcebm"}
)
SPATIAL_POTTS_MIXING_SCHEMES = frozenset({"probability", "energy"})


@dataclass(frozen=True)
class PottsLocalMixture:
    """Secondary Potts branch and composition rule for local DLMC mixing."""

    h: torch.Tensor
    J: torch.Tensor
    edge_idx: torch.LongTensor
    mask_i: torch.Tensor
    alpha: torch.Tensor | None
    mixing_scheme: PottsMixingScheme


@dataclass(frozen=True)
class PottsObjective:
    """One additional objective in a multi-objective Potts composition."""

    label: str
    h: torch.Tensor
    J: torch.Tensor
    edge_idx: torch.LongTensor
    mask_i: torch.Tensor


@dataclass(frozen=True)
class PottsMultiObjectiveMixture:
    """Additional branches for simplex-constrained multi-objective pcEBM."""

    primary_label: str
    additional_objectives: tuple[PottsObjective, ...]
    mixing_scheme: Literal["pcebm_multi"] = "pcebm_multi"


@dataclass(frozen=True)
class PottsMixingDiagnostics:
    """Per-sequence pcEBM pocket weights accumulated during sampling."""

    lambda_pocket_final: torch.Tensor
    lambda_pocket_min: torch.Tensor
    lambda_pocket_max: torch.Tensor
    lambda_pocket_mean: torch.Tensor


@dataclass(frozen=True)
class PottsMultiObjectiveDiagnostics:
    """Per-objective pcEBM weights accumulated during sampling."""

    objective_labels: tuple[str, ...]
    objective_weights_last_proposal: torch.Tensor
    objective_weights_terminal: torch.Tensor
    objective_weights_min: torch.Tensor
    objective_weights_max: torch.Tensor
    objective_weights_mean: torch.Tensor


@dataclass(frozen=True)
class _PottsProposalResult:
    """Energy, transition logits, and optional mixing weight for one proposal."""

    energy: torch.Tensor
    log_probs: torch.Tensor
    lambda_pocket: torch.Tensor | None = None
    objective_weights: torch.Tensor | None = None


class GraphPotts(nn.Module):
    """Conditional Random Field (conditional Potts model) layer on a graph.

    Arguments:
        dim_nodes (int): Hidden dimension of node tensor.
        dim_edges (int): Hidden dimension of edge tensor.
        num_states (int): Size of the vocabulary.
        parameterization (str): Either the single-head `factor` model or the
            shared-edge gated `multi_head_factor` model.
        symmetric_J (bool): If True enforce symmetry of Potts model i.e.
            `J_ij(s_i, s_j) = J_ji(s_j, s_i)`.
        init_scale (float): Scale factor for the weights and couplings at
            initialization.
        dropout (float): Probability of per-dimension dropout on `[0,1]`.
        num_heads (int): Number of shared-edge Potts heads.
        reduce (str): Shared-edge head reduction; currently `mean`.

    Inputs:
        node_h (torch.Tensor): Node features with shape
            `(num_batch, num_nodes, dim_nodes)`.
        edge_h (torch.Tensor): Edge features with shape
            `(num_batch, num_nodes, num_neighbors, dim_edges)`.
        edge_idx (torch.LongTensor): Edge indices with shape
            `(num_batch, num_nodes, num_neighbors)`.
        mask_i (torch.Tensor): Node mask with shape `(num_batch, num_nodes)`
        mask_ij (torch.Tensor): Edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`

    Outputs:
        h (torch.Tensor): Potts model fields :math:`h_i(s_i)` with shape
            `(num_batch, num_nodes, num_states)`.
        J (Tensor): Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
            `(num_batch, num_nodes, num_neighbors, num_states, num_states)`.
    """

    def __init__(
        self,
        dim_nodes: int,
        dim_edges: int,
        num_states: int,
        parameterization: str = "factor",
        symmetric_J: bool = True,
        init_scale: float = 0.1,
        dropout: float = 0.0,
        num_heads: Optional[int] = None,
        reduce: str = "mean",
        full_multi_head_aggregation: Optional[str] = None,
        adapter_hidden_dim: Optional[int] = None,
        shared_edge_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim_nodes = dim_nodes
        self.dim_edges = dim_edges
        self.num_states = num_states
        self.init_scale = init_scale
        self.parameterization = parameterization
        self.symmetric_J = symmetric_J
        self.full_multi_head_aggregation = full_multi_head_aggregation
        self.shared_edge_dim = shared_edge_dim

        if self.parameterization == "factor":
            if (
                self.full_multi_head_aggregation is not None
                or self.shared_edge_dim is not None
            ):
                raise ValueError(
                    "factor Potts does not use multi-head aggregation or "
                    "shared_edge_dim"
                )
            self.log_scale = nn.Parameter(
                np.log(init_scale) * torch.ones(1)
            )
            self.W_h = nn.Linear(
                self.dim_nodes,
                self.num_states,
                bias=True,
            )
            self.W_J_left = nn.Linear(
                self.dim_edges,
                self.num_states**2,
                bias=True,
            )
            self.W_J_right = nn.Linear(
                self.dim_edges,
                self.num_states**2,
                bias=True,
            )
        elif self.parameterization == "multi_head_factor":
            if self.full_multi_head_aggregation != "gate_nonlinear":
                raise ValueError(
                    "multi_head_factor requires "
                    "full_multi_head_aggregation='gate_nonlinear'"
                )
            if reduce != "mean":
                raise ValueError(
                    "shared-edge multi-head Potts requires reduce='mean'; "
                    f"got {reduce!r}"
                )
            if num_heads is None:
                raise ValueError(
                    "shared-edge multi-head Potts requires num_heads"
                )
            if adapter_hidden_dim is None:
                raise ValueError(
                    "shared-edge multi-head Potts requires adapter_hidden_dim"
                )
            if isinstance(shared_edge_dim, bool) or not isinstance(
                shared_edge_dim,
                int,
            ):
                raise ValueError(
                    "shared-edge multi-head Potts requires integer "
                    f"shared_edge_dim; got {shared_edge_dim!r}"
                )

            self.num_heads = int(num_heads)
            self.reduce = reduce
            self.full_multi_head_factor = (
                SharedEdgeBottleneckMultiHeadFactorPotts(
                    dim_nodes=self.dim_nodes,
                    dim_edges=self.dim_edges,
                    shared_edge_dim=shared_edge_dim,
                    num_states=self.num_states,
                    num_heads=self.num_heads,
                    node_adapter_hidden_dim=adapter_hidden_dim,
                    init_scale=init_scale,
                    dropout=dropout,
                )
            )
        else:
            raise ValueError(
                "Unsupported Potts parameterization: "
                f"{self.parameterization!r}. Expected 'factor' or "
                "'multi_head_factor'."
            )
        self.dropout = nn.Dropout(dropout)

    def reset_full_multi_head_special_initialization(self) -> None:
        if (
            self.parameterization == "multi_head_factor"
            and self.full_multi_head_factor is not None
        ):
            self.full_multi_head_factor.reset_special_initialization()


    def forward(
        self,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        edge_idx: torch.LongTensor,
        mask_i: torch.Tensor,
        mask_ij: torch.Tensor,
        return_multi_head_stats: bool = False,
    ):
        #! (JH) 260131Note
        # edge_idx: E_idx between only protein tokens in protein chains,
        # mask_i: protein_residue_node_mask, mask_ij: protein_residue_edge_mask_2d
        mask_J = build_coupling_mask(
            edge_idx,
            mask_i,
            mask_ij,
            require_reciprocal=False,
        )
        multi_head_stats = None

        if self.parameterization == "factor":
            scale = torch.exp(self.log_scale)
            h = scale * mask_i.unsqueeze(-1) * self.W_h(node_h)
            mask_J = scale * mask_J.unsqueeze(-1)
            shape_J = list(edge_h.size())[:3] + ([self.num_states] * 2)
            J_left = (mask_J * self.W_J_left(edge_h)).view(shape_J)
            J_right = (mask_J * self.W_J_right(edge_h)).view(shape_J)
            J = torch.matmul(J_left, J_right)
            J = self.dropout(J)
            h = h - h.mean(-1, keepdim=True)
            J = (
                J
                - J.mean(-1, keepdim=True)
                - J.mean(-2, keepdim=True)
                + J.mean(dim=[-1, -2], keepdim=True)
            )
        else:
            output = self.full_multi_head_factor(
                node_h,
                edge_h,
                mask_i,
                mask_J,
                return_stats=return_multi_head_stats,
            )
            if return_multi_head_stats:
                h, J, multi_head_stats = output
            else:
                h, J = output

        if self.symmetric_J:
            J = self._symmetrize_J(J, edge_idx, mask_ij)

        if return_multi_head_stats:
            return h, J, multi_head_stats
        return h, J

    def _symmetrize_J_serial(self, J, edge_idx, mask_ij):
        """Enforce symmetry of J matrices, serial version."""
        num_batch, num_residues, num_k, num_states, _ = list(J.size())

        # Symmetrization based on raw indexing - extremely slow; for debugging
        import time

        _start = time.time()
        J_symm = torch.zeros_like(J)
        for b in range(J.size(0)):
            for i in range(J.size(1)):
                for k_i in range(J.size(2)):
                    for k_j in range(J.size(2)):
                        j = edge_idx[b, i, k_i]
                        if edge_idx[b, j, k_j] == i:
                            J_symm[b, i, k_i, :, :] = (
                                J[b, i, k_i, :, :]
                                + J[b, j, k_j, :, :].transpose(-1, -2)
                            ) / 2.0
        speed = J.size(0) * J.size(1) / (time.time() - _start)
        print(f"symmetrized at {speed} residue/s")
        return J_symm

    def _symmetrize_J(self, J, edge_idx, mask_ij):
        """Enforce symmetry of J matrices via adding J_ij + J_ji^T"""
        num_batch, num_residues, num_k, num_states, _ = list(J.size())

        # Flatten and gather J_ji matrices using transpose indexing
        J_flat = J.view(num_batch, num_residues, num_k, -1)
        J_flat_transpose, mask_ji = graph.collect_edges_transpose(
            J_flat, edge_idx, mask_ij
        )
        J_transpose = J_flat_transpose.view(
            num_batch, num_residues, num_k, num_states, num_states
        )
        # Transpose J_ji matrices to symmetrize as (J_ij + J_ji^T)/2
        J_transpose = J_transpose.transpose(-2, -1)
        mask_ji = (0.5 * mask_ji).view(num_batch, num_residues, num_k, 1, 1)
        J_symm = mask_ji * (J + J_transpose)
        return J_symm

    def energy(
        self,
        S: torch.LongTensor,
        h: torch.Tensor,
        J: torch.Tensor,
        edge_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """Compute Potts model energy from sequence.

        Inputs:
            S (torch.LongTensor): Sequence with shape `(num_batch, num_nodes)`.
            h (torch.Tensor): Potts model fields :math:`h_i(s_i)` with shape
                `(num_batch, num_nodes, num_states)`.
            J (Tensor): Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
                `(num_batch, num_nodes, num_neighbors, num_states, num_states)`.
            edge_idx (torch.LongTensor): Edge indices with shape
                `(num_batch, num_nodes, num_neighbors)`.

        Outputs:
            U (torch.Tensor): Potts total energies with shape `(num_batch)`.
                Lower energies are more favorable.
        """
        # Gather J [Batch,i,j,A_i,A_j] => J_ij(:,A_j) [Batch,i,j,A_i]
        S_j = graph.collect_neighbors(S.unsqueeze(-1), edge_idx)
        S_j = S_j.unsqueeze(-1).expand(-1, -1, -1, self.num_states, -1)
        J_ij = torch.gather(J, -1, S_j).squeeze(-1)

        # Sum out J contributions
        J_i = J_ij.sum(2) / 2.0
        r_i = h + J_i

        U_i = torch.gather(r_i, 2, S.unsqueeze(-1))
        U = U_i.sum([1, 2])
        return U

    def pseudolikelihood(
        self,
        S: torch.LongTensor,
        h: torch.Tensor,
        J: torch.Tensor,
        edge_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """Compute Potts pseudolikelihood from sequence

        Inputs:
            S (torch.LongTensor): Sequence with shape `(num_batch, num_nodes)`.
            h (torch.Tensor): Potts model fields :math:`h_i(s_i)` with shape
                `(num_batch, num_nodes, num_states)`.
            J (Tensor): Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
                `(num_batch, num_nodes, num_neighbors, num_states, num_states)`.
            edge_idx (torch.LongTensor): Edge indices with shape
                `(num_batch, num_nodes, num_neighbors)`.

        Outputs:
            log_probs (torch.Tensor): Potts log-pseudolihoods with shape
                `(num_batch, num_nodes, num_states)`.
        """
        return pseudolikelihood(S, h, J, edge_idx)


    def sample(
        self,
        h: torch.Tensor,
        J: torch.Tensor,
        edge_idx: torch.LongTensor,
        mask_i: torch.Tensor,
        mask_ij: torch.Tensor,
        S: Optional[torch.LongTensor] = None,
        mask_sample: Optional[torch.Tensor] = None,
        num_sweeps: int = 100,
        temperature: float = 0.1,
        temperature_init: float = 1.0,
        penalty_func: Optional[Callable[[torch.LongTensor], torch.Tensor]] = None,
        differentiable_penalty: bool = True,
        rejection_step: bool = False,
        proposal: Literal["dlmc", "chromatic"] = "dlmc",
        verbose: bool = False,
        edge_idx_coloring: Optional[torch.LongTensor] = None,
        mask_ij_coloring: Optional[torch.Tensor] = None,
        symmetry_order: Optional[int] = None,
        h_uncond: Optional[torch.Tensor] = None,
        J_uncond: Optional[torch.Tensor] = None,
        edge_idx_uncond: Optional[torch.LongTensor] = None,
        gamma: float = 1.0,
        gamma_schedule_cfg: Optional[dict] = None,
        local_mixture: Optional[
            Union[PottsLocalMixture, PottsMultiObjectiveMixture]
        ] = None,
        temperature_schedule_cfg: Optional[dict] = None,
        temperature_schedule_override: Optional[torch.Tensor] = None,
        return_temperature_schedule: bool = False,
        return_mixing_diagnostics: bool = False,
        dlmc_dt: float = 0.1,
    ) -> Union[
        tuple[torch.LongTensor, torch.Tensor],
        tuple[torch.LongTensor, torch.Tensor, InferenceScheduleTrace],
        tuple[torch.LongTensor, torch.Tensor, PottsMixingDiagnostics],
        tuple[torch.LongTensor, torch.Tensor, PottsMultiObjectiveDiagnostics],
    ]:
        """Sample from Potts model with Chromatic Gibbs sampling.

        Args:
            h: Potts model fields :math:`h_i(s_i)` with shape
                `(num_batch, num_nodes, num_states)`.
            J: Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
                `(num_batch, num_nodes, num_neighbors, num_states, num_states)`.
            edge_idx (torch.LongTensor): Edge indices with shape
                `(num_batch, num_nodes, num_neighbors)`.
            mask_i (torch.Tensor): Node mask with shape `(num_batch, num_nodes)`.
            mask_ij (torch.Tensor): Edge mask with shape
                `(num_batch, num_nodes, num_neighbors)`.
            S (torch.LongTensor, optional): Sequence for initialization with
                shape `(num_batch, num_nodes)`.
            mask_sample (torch.Tensor, optional): Binary sampling mask indicating
                positions which are free to change with shape
                `(num_batch, num_nodes)` or which tokens are acceptable at each position
                with shape `(num_batch, num_nodes, alphabet)`.
            num_sweeps (int): Number of sweeps of Chromatic Gibbs to perform,
                i.e. the depth of sampling as measured by the number of times
                every position has had an opportunity to update.
            temperature (float): Final sampling temperature.
            temperature_init (float): Initial sampling temperature, which will
                be linearly interpolated to `temperature` over the course of
                the burn in phase.
            penalty_func (Callable, optional): An optional penalty function which
                takes a sequence `S` and outputes a `(num_batch)` shaped tensor
                of energy adjustments, for example as regularization.
            differentiable_penalty (bool): If True, gradients of penalty function
                will be used to adjust the proposals.
            rejection_step (bool): If True, perform a Metropolis-Hastings
                rejection step.
            proposal (str): MCMC proposal for Potts sampling. Currently implemented
                proposals are `dlmc` for Discrete Langevin Monte Carlo [1] or `chromatic`
                for Gibbs sampling with graph coloring.
                [1] Sun et al. Discrete Langevin Sampler via Wasserstein Gradient Flow (2023).
            dlmc_dt (float): Step size used by DLMC transition probabilities. Ignored
                when ``proposal="chromatic"``.
            verbose (bool): If True print verbose output during sampling.
            edge_idx_coloring (torch.LongerTensor, optional): Alternative
                graph dependency structure that can be provided for the
                Chromatic Gibbs algorithm when it performs initial graph
                coloring. Has shape
                    `(num_batch, num_nodes, num_neighbors_coloring)`.
            mask_ij_coloring (torch.Tensor): Edge mask for the alternative dependency
                structure with shape `(num_batch, num_nodes, num_neighbors_coloring)`.
            symmetry_order (int, optional): Optional integer argument to enable
                symmetric sequence decoding under `symmetry_order`-order symmetry.
                The first `(num_nodes // symmetry_order)` states will be free to
                move, and all consecutively tiled sets of states will be locked
                to these during decoding. Internally this is accomplished by
                summing the parameters Potts model under a symmetry constraint
                into this reduced sized system and then back imputing at the end.

        Returns:
            S (torch.LongTensor): Sampled sequences with
                shape `(num_batch, num_nodes)`.
            U (torch.Tensor): Sampled energies with shape `(num_batch)`. Lower
                is more favorable.
        """
        B, N, _ = h.shape

        if symmetry_order is not None:
            schedule_enabled = bool(temperature_schedule_cfg) and bool(
                temperature_schedule_cfg.get("enabled", True)
            )
            if schedule_enabled or temperature_schedule_override is not None:
                raise NotImplementedError(
                    "inference temperature schedules do not support symmetry folding"
                )
            if h_uncond is not None or local_mixture is not None:
                raise NotImplementedError(
                    "symmetry_order cannot be combined with a second Potts branch."
                )
            h, J, edge_idx, mask_i, mask_ij = fold_symmetry(
                symmetry_order, h, J, edge_idx, mask_i, mask_ij
            )
            S = S[:, : (N // symmetry_order)]
            if mask_sample is not None:
                mask_sample = mask_sample[:, : (N // symmetry_order)]

        sample_output = sample_potts(
            h,
            J,
            edge_idx,
            mask_i,
            mask_ij,
            S=S,
            mask_sample=mask_sample,
            num_sweeps=num_sweeps,
            temperature=temperature,
            temperature_init=temperature_init,
            penalty_func=penalty_func,
            differentiable_penalty=differentiable_penalty,
            rejection_step=rejection_step,
            proposal=proposal,
            dlmc_dt=dlmc_dt,
            verbose=verbose,
            edge_idx_coloring=edge_idx_coloring,
            mask_ij_coloring=mask_ij_coloring,
            h_uncond=h_uncond,
            J_uncond=J_uncond,
            edge_idx_uncond=edge_idx_uncond,
            gamma=gamma,
            gamma_schedule_cfg=gamma_schedule_cfg,
            local_mixture=local_mixture,
            temperature_schedule_cfg=temperature_schedule_cfg,
            temperature_schedule_override=temperature_schedule_override,
            return_temperature_schedule=return_temperature_schedule,
            return_mixing_diagnostics=return_mixing_diagnostics,
        )
        if return_temperature_schedule:
            S_sample, U_sample, temperature_schedule_trace = sample_output
        elif return_mixing_diagnostics:
            S_sample, U_sample, mixing_diagnostics = sample_output
        else:
            S_sample, U_sample = sample_output

        if symmetry_order is not None:
            assert N % symmetry_order == 0
            S_sample = (
                S_sample[:, None, :].expand([-1, symmetry_order, -1]).reshape([B, N])
            )
        if return_temperature_schedule:
            return S_sample, U_sample, temperature_schedule_trace
        if return_mixing_diagnostics:
            return S_sample, U_sample, mixing_diagnostics
        return S_sample, U_sample


def compute_potts_energy(
    S: torch.LongTensor,
    h: torch.Tensor,
    J: torch.Tensor,
    edge_idx: torch.LongTensor,
    return_per_res: bool = False,
):
    """Compute Potts model energies from sequence.

    Args:
        S (torch.LongTensor): Sequence with shape `(num_batch, num_nodes)`.
        h (torch.Tensor): Potts model fields :math:`h_i(s_i)` with shape
            `(num_batch, num_nodes, num_states)`.
        J (Tensor): Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
            `(num_batch, num_nodes, num_neighbors, num_states, num_states)`.
        edge_idx (torch.LongTensor): Edge indices with shape
            `(num_batch, num_nodes, num_neighbors)`.
        return_per_res (bool): If True, additionally return the per-residue
            contributions ``U_per_res`` of shape `(num_batch, num_nodes)` that
            sum along the node axis to ``U``.

    Returns:
        U (torch.Tensor): Potts total energies with shape `(num_batch)`.
            Lower energies are more favorable.
        U_i (torch.Tensor): Potts local conditional energies with shape
            `(num_batch, num_nodes, num_states)`.
        U_per_res (torch.Tensor, optional): Per-residue contributions with
            shape `(num_batch, num_nodes)`. Only returned when
            ``return_per_res=True``.
    """
    S_j = graph.collect_neighbors(S.unsqueeze(-1), edge_idx) # S: [b, n] / S_j: [b, n, k, 1]
    # S_j: neighbor's state
    S_j = S_j.unsqueeze(-1).expand(-1, -1, -1, h.shape[-1], -1)
    # S_j: [b, n, k, num_states, 1]. but the second last dimension is just copied num_states times
    J_ij = torch.gather(J, -1, S_j).squeeze(-1)
    # J: [b, n, k, num_states, num_states]
    # J_ij: Along the last axis, select only the column indicated by S_j at each position i,
    # yielding a tensor of shape (B, N, K, Q, 1) -> (B, N, K, Q)

    # Sum out J contributions to yield local conditionals
    J_i = J_ij.sum(2) # sum over neighbors, J_i: [b, n, num_states]
    U_i = h + J_i # U_i: [b, n, num_states]

    # Per-residue contribution: h_i(S_i) + 0.5 * sum_j J_{ij}(S_i, S_j).
    # The 0.5 corrects for double counting of each edge across the two endpoints.
    U_per_res = (
        torch.gather(U_i, -1, S[..., None]).squeeze(-1)
        - 0.5 * torch.gather(J_i, -1, S[..., None]).squeeze(-1)
    )  # [b, n]
    U = U_per_res.sum(-1)  # [b]
    if return_per_res:
        return U, U_i, U_per_res
    return U, U_i


def fold_symmetry(
    symmetry_order: int,
    h: torch.Tensor,
    J: torch.Tensor,
    edge_idx: torch.LongTensor,
    mask_i: torch.Tensor,
    mask_ij: torch.Tensor,
    normalize=True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fold Potts model symmetrically.

    Args:
        symmetry_order (int): The order of symmetry by which to fold the Potts
            model such that the first `(num_nodes // symmetry_order)` states
            represent the entire system and all fields and couplings to and
            among other copies of this base system are collected together in
            single reduced Potts model.
        h (torch.Tensor): Potts model fields :math:`h_i(s_i)` with shape
            `(num_batch, num_nodes, num_states)`.
        J (Tensor): Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
            `(num_batch, num_nodes, num_neighbors, num_states, num_states)`.
        edge_idx (torch.LongTensor): Edge indices with shape
            `(num_batch, num_nodes, num_neighbors)`.
        mask_i (torch.Tensor): Node mask with shape `(num_batch, num_nodes)`.
        mask_ij (torch.Tensor): Edge mask with shape
            `(num_batch, num_nodes, num_neighbors)`.
        normalize (bool): If True (default), aggregate the Potts model as an average
            energy across asymmetric units instead of as a sum.

    Returns:
        h_fold (torch.Tensor): Potts model fields :math:`h_i(s_i)` with shape
            `(num_batch, num_nodes_folded, num_states)`, where
            `num_nodes_folded =  num_nodes // symmetry_order`.
        J_fold (Tensor): Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
            `(num_batch, num_nodes_folded, num_neighbors, num_states, num_states)`.
        edge_idx_fold (torch.LongTensor): Edge indices with shape
            `(num_batch, num_nodes_folded, num_neighbors)`.
        mask_i_fold (torch.Tensor): Node mask with shape `(num_batch, num_nodes_folded)`.
        mask_ij_fold (torch.Tensor): Edge mask with shape
            `(num_batch, num_nodes_folded, num_neighbors)`.

    """
    B, N, K, Q, _ = J.shape
    device = h.device

    N_asymmetric = N // symmetry_order
    # Fold edges by densifying the assymetric unit and averaging
    edge_idx_au = torch.remainder(edge_idx, N_asymmetric).clamp(max=N_asymmetric - 1)

    def _pairwise_fold(_T):
        # Fold-sum along neighbor dimension
        shape = list(_T.shape)
        shape[2] = N_asymmetric
        _T_au_expand = torch.zeros(shape, device=device).float()
        extra_dims = len(_T.shape) - len(edge_idx_au.shape)
        edge_idx_au_expand = edge_idx_au.reshape(
            list(edge_idx_au.shape) + [1] * extra_dims
        ).expand([-1, -1, -1] + [Q] * extra_dims)
        _T_au_expand.scatter_add_(2, edge_idx_au_expand, _T.float())

        # Fold-mean along self dimension
        shape_out = [shape[0], -1, N_asymmetric, N_asymmetric] + shape[3:]
        _T_au = _T_au_expand.reshape(shape_out).sum(1)
        return _T_au

    J_fold = _pairwise_fold(J)
    mask_ij_fold = (_pairwise_fold(mask_ij) > 0).float()
    edge_idx_fold = (
        torch.arange(N_asymmetric, device=device)
        .long()[None, None, :]
        .expand(mask_ij_fold.shape)
    )

    # Drop unused edges
    K_fold = mask_ij_fold.sum(2).max().item()
    _, sort_ix = torch.sort(mask_ij_fold, dim=2, descending=True)
    sort_ix_J = sort_ix[..., None, None].expand(list(sort_ix.shape) + [Q, Q])
    edge_idx_fold = torch.gather(edge_idx_fold, 2, sort_ix)
    mask_ij_fold = torch.gather(mask_ij_fold, 2, sort_ix)
    J_fold = torch.gather(J_fold, 2, sort_ix_J)

    # Fold-mean along self dimension
    h_fold = h.reshape([B, -1, N_asymmetric, Q]).sum(1)
    mask_i_fold = (mask_i.reshape([B, -1, N_asymmetric]).sum(1) > 0).float()
    if normalize:
        h_fold = h_fold / symmetry_order
        J_fold = J_fold / symmetry_order
    return h_fold, J_fold, edge_idx_fold, mask_i_fold, mask_ij_fold


@torch.no_grad()
def _color_graph(edge_idx, mask_ij, max_iter=100):
    """Stochastic graph coloring."""
    # Randomly assign initial colors
    B, N, K = edge_idx.shape
    # By Brooks we only need K + 1, but one extra color aids convergence
    num_colors = K + 2
    S = torch.randint(0, num_colors, (B, N), device=edge_idx.device)

    # Ignore self-attachement
    ix = torch.arange(edge_idx.shape[1], device=edge_idx.device)[None, ..., None]
    mask_ij = (mask_ij * torch.ne(edge_idx, ix).float())[..., None]

    # Iteratively replace clashing sites with an available color
    i = 0
    total_clashes = 1
    while total_clashes > 0 and i < max_iter:
        # Tabulate available colors in neighborhood
        O_i = F.one_hot(S, num_colors).float()
        N_i = (mask_ij * graph.collect_neighbors(O_i, edge_idx)).sum(2)
        clashes = (O_i * N_i).sum(-1)
        N_i = torch.where(N_i > 0, -float("inf") * torch.ones_like(N_i), N_i)

        # Resample from this distribution where clashing
        S_new = torch.distributions.categorical.Categorical(logits=N_i).sample()
        S = torch.where(clashes > 0, S_new, S)
        i += 1
        total_clashes = clashes.sum().item()
    return S


@torch.no_grad()
def _build_gamma_schedule(num_iterations: int, schedule_cfg: dict) -> List[float]:
    """Return γ at every DLMC step for a time-dependent guidance schedule.

    Convention: t = 1 - k / max(1, num_iterations - 1), so k=0 → t=1 (initial
    / fully masked) and k=num_iterations-1 → t=0 (final). Schedules from
    Rojas et al., ICLR 2026, Table 2.

    Supported types:
        - "constant":       γ(t) = gamma_max
        - "ramp_up":        γ(t) = min(gamma_max, gamma_max * (1 - t) / (1 - tau))
        - "right_interval": γ(t) = gamma_max if t >= tau else 0.0
    """
    N = int(num_iterations)
    if N <= 0:
        return []
    g_max = float(schedule_cfg["gamma_max"])
    stype = schedule_cfg["type"]
    denom_k = max(1, N - 1)

    out: List[float] = []
    for k in range(N):
        t = 1.0 - k / denom_k
        if stype == "constant":
            g = g_max
        elif stype == "ramp_up":
            tau = float(schedule_cfg["tau"])
            denom = max(1e-12, 1.0 - tau)
            g = min(g_max, g_max * (1.0 - t) / denom)
        elif stype == "right_interval":
            tau = float(schedule_cfg["tau"])
            g = g_max if t >= tau else 0.0
        else:
            raise ValueError(f"Unknown gamma schedule type: {stype!r}")
        out.append(float(g))
    return out


def sample_potts(
    h: torch.Tensor,
    J: torch.Tensor,
    edge_idx: torch.LongTensor,
    mask_i: torch.Tensor,
    mask_ij: torch.Tensor,
    S: Optional[torch.LongTensor] = None,
    mask_sample: Optional[torch.Tensor] = None,
    num_sweeps: int = 100,
    temperature: float = 1.0,
    temperature_init: float = 1.0,
    annealing_fraction: float = 0.8,
    penalty_func: Optional[Callable[[torch.LongTensor], torch.Tensor]] = None,
    differentiable_penalty: bool = True,
    rejection_step: bool = False,
    proposal: Literal["dlmc", "chromatic"] = "dlmc",
    verbose: bool = True,
    return_trajectory: bool = False,
    thin_sweeps: int = 3,
    edge_idx_coloring: Optional[torch.LongTensor] = None,
    mask_ij_coloring: Optional[torch.Tensor] = None,
    h_uncond: Optional[torch.Tensor] = None,
    J_uncond: Optional[torch.Tensor] = None,
    edge_idx_uncond: Optional[torch.LongTensor] = None,
    gamma: float = 1.0,
    gamma_schedule_cfg: Optional[dict] = None,
    local_mixture: Optional[
        Union[PottsLocalMixture, PottsMultiObjectiveMixture]
    ] = None,
    temperature_schedule_cfg: Optional[dict] = None,
    temperature_schedule_override: Optional[torch.Tensor] = None,
    return_temperature_schedule: bool = False,
    return_mixing_diagnostics: bool = False,
    dlmc_dt: float = 0.1,
) -> Union[
    tuple[torch.LongTensor, torch.Tensor],
    tuple[torch.LongTensor, torch.Tensor, InferenceScheduleTrace],
    tuple[torch.LongTensor, torch.Tensor, PottsMixingDiagnostics],
    tuple[torch.LongTensor, torch.Tensor, PottsMultiObjectiveDiagnostics],
    tuple[torch.LongTensor, torch.Tensor, list[torch.LongTensor], list[torch.Tensor]],
]:
    """Sample from Potts model with Chromatic Gibbs sampling.

    Args:
        h (torch.Tensor): Potts model fields :math:`h_i(s_i)` with shape
            `(num_batch, num_nodes, num_states)`.
        J (Tensor): Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
            `(num_batch, num_nodes, num_neighbors, num_states, num_states)`.
        edge_idx (torch.LongTensor): Edge indices with shape
            `(num_batch, num_nodes, num_neighbors)`.
        mask_i (torch.Tensor): Node mask with shape `(num_batch, num_nodes)`.
        mask_ij (torch.Tensor): Edge mask with shape
            `(num_batch, num_nodes, num_neighbors)`.
        S (torch.LongTensor, optional): Sequence for initialization with
            shape `(num_batch, num_nodes)`.
        mask_sample (torch.Tensor, optional): Binary sampling mask indicating
            positions which are free to change with shape
            `(num_batch, num_nodes)` or which tokens are acceptable at each position
            with shape `(num_batch, num_nodes, alphabet)`.
        num_sweeps (int): Number of sweeps of Chromatic Gibbs to perform,
            i.e. the depth of sampling as measured by the number of times
            every position has had an opportunity to update.
        temperature (float): Final sampling temperature.
        temperature_init (float): Initial sampling temperature, which will
            be linearly interpolated to `temperature` over the course of
            the burn in phase.
        annealing_fraction (float): Fraction of the total sampling run during
            which temperature annealing occurs.
        penalty_func (Callable, optional): An optional penalty function which
            takes a sequence `S` and outputes a `(num_batch)` shaped tensor
            of energy adjustments, for example as regularization.
        differentiable_penalty (bool): If True, gradients of penalty function
            will be used to adjust the proposals.
        rejection_step (bool): If True, perform a Metropolis-Hastings
            rejection step.
        proposal (str): MCMC proposal for Potts sampling. Currently implemented
                proposals are `dlmc` for Discrete Langevin Monte Carlo [1] or `chromatic`
                for Gibbs sampling with graph coloring.
                [1] Sun et al. Discrete Langevin Sampler via Wasserstein Gradient Flow (2023).
        dlmc_dt (float): Step size used by DLMC transition probabilities. Ignored
            when ``proposal="chromatic"``.
        verbose (bool): If True print verbose output during sampling.
        return_trajectory (bool): If True, also output the sampling trajectories
            of `S` and `U`.
        thin_sweeps (int): When returning trajectories, only save every `thin_sweeps`
            state to reduce memory usage.
        edge_idx_coloring (torch.LongerTensor, optional): Alternative
            graph dependency structure that can be provided for the
            Chromatic Gibbs algorithm when it performs initial graph
            coloring. Has shape
                `(num_batch, num_nodes, num_neighbors_coloring)`.
        mask_ij_coloring (torch.Tensor): Edge mask for the alternative dependency
            structure with shape `(num_batch, num_nodes, num_neighbors_coloring)`.

    Returns:
        S (torch.LongTensor): Sampled sequences with
            shape `(num_batch, num_nodes)`.
        U (torch.Tensor): Sampled energies with shape `(num_batch)`. Lower is more
            favorable.atb
        S_trajectory (list[torch.LongTensor]): List of sampled sequences through
            time each with shape `(num_batch, num_nodes)`.
        U_trajectory (list[torch.Tensor]): List of sampled energies through time
            each with shape `(num_batch)`.
    """
    # Initialize masked proposals and mask h
    mask_S, mask_mutatable, S = init_sampling_masks(-h, mask_sample, S) # mask_mutatable is mask_S_1D
    legal_state_mask = (
        (mask_S > 0)
        & (mask_mutatable[..., None] > 0)
        & (mask_i[..., None] > 0)
    )
    h_numerical_zero = h.max() + 1e3 * max(1.0, temperature) # Prohibit sampling tokens where mask_S > 0
    h = torch.where(mask_S > 0, h, h_numerical_zero * torch.ones_like(h))

    # Classifier-free-style guidance: if an uncond branch is provided, we
    # sample from a mix of the cond and uncond DLMC proposals at every sweep.
    use_guidance = h_uncond is not None
    use_local_mixture = local_mixture is not None
    proposal_mode = resolve_potts_proposal_mode(
        proposal,
        has_guidance=use_guidance,
        has_local_mixture=use_local_mixture,
        rejection_step=rejection_step,
    )
    if return_mixing_diagnostics:
        if local_mixture is None or local_mixture.mixing_scheme not in {
            "pcebm",
            "pcebm_multi",
        }:
            raise ValueError(
                "return_mixing_diagnostics requires mixing_scheme='pcebm' "
                "or 'pcebm_multi'"
            )
        if return_temperature_schedule or return_trajectory:
            raise NotImplementedError(
                "pcEBM mixing diagnostics cannot be combined with temperature "
                "schedule traces or state trajectories"
            )
    if use_local_mixture:
        if isinstance(local_mixture, PottsMultiObjectiveMixture):
            local_mixture = _validate_multi_objective_mixture(
                local_mixture,
                h,
                mask_i,
            )
            masked_objectives = []
            for objective in local_mixture.additional_objectives:
                h_objective_numerical_zero = objective.h.max() + 1e3 * max(
                    1.0, temperature
                )
                masked_objectives.append(
                    PottsObjective(
                        label=objective.label,
                        h=torch.where(
                            mask_S > 0,
                            objective.h,
                            h_objective_numerical_zero
                            * torch.ones_like(objective.h),
                        ),
                        J=objective.J,
                        edge_idx=objective.edge_idx,
                        mask_i=objective.mask_i,
                    )
                )
            local_mixture = PottsMultiObjectiveMixture(
                primary_label=local_mixture.primary_label,
                additional_objectives=tuple(masked_objectives),
            )
        else:
            local_mixture = _validate_local_mixture(local_mixture, h, mask_i)
            h_secondary_numerical_zero = local_mixture.h.max() + 1e3 * max(
                1.0, temperature
            )
            local_mixture = PottsLocalMixture(
                h=torch.where(
                    mask_S > 0,
                    local_mixture.h,
                    h_secondary_numerical_zero * torch.ones_like(local_mixture.h),
                ),
                J=local_mixture.J,
                edge_idx=local_mixture.edge_idx,
                mask_i=local_mixture.mask_i,
                alpha=local_mixture.alpha,
                mixing_scheme=local_mixture.mixing_scheme,
            )
    if use_guidance:
        assert J_uncond is not None and edge_idx_uncond is not None, (
            "h_uncond was provided but J_uncond / edge_idx_uncond are missing."
        )
        # Apply a per-branch numerical-zero floor. Reusing cond's floor
        # (h.max() + 1e3*T) under-suppresses banned tokens whenever
        # h_uncond.max() > h.max(), which in turn widens the cond/uncond
        # disagreement that CFG at gamma > 1 amplifies. Computing the
        # uncond floor from h_uncond.max() keeps banned = numerically
        # zero in both branches regardless of their relative scale.
        h_numerical_zero_uncond = h_uncond.max() + 1e3 * max(1.0, temperature)
        h_uncond = torch.where(
            mask_S > 0,
            h_uncond,
            h_numerical_zero_uncond * torch.ones_like(h_uncond),
        )

    # Block update schedule
    if proposal_mode is PottsProposalMode.CHROMATIC:
        if edge_idx_coloring is None:
            edge_idx_coloring = edge_idx
        if mask_ij_coloring is None:
            mask_ij_coloring = mask_ij
        schedule = _color_graph(edge_idx_coloring, mask_ij_coloring)
        num_colors = schedule.max() + 1
        num_iterations = num_colors * num_sweeps
    else:
        num_iterations = num_sweeps

    temperature_schedule_trace = None
    schedule_enabled = bool(temperature_schedule_cfg) and bool(
        temperature_schedule_cfg.get("enabled", True)
    )
    if temperature_schedule_override is not None and not schedule_enabled:
        raise ValueError(
            "temperature_schedule_override requires an enabled temperature schedule"
        )
    if schedule_enabled:
        use_mean_energy_mixture_schedule = (
            proposal_mode is PottsProposalMode.LOCAL_MIXTURE_DLMC
            and local_mixture is not None
            and local_mixture.mixing_scheme == "mean_energy"
        )
        if (
            proposal_mode is not PottsProposalMode.DLMC
            and not use_mean_energy_mixture_schedule
        ):
            raise NotImplementedError(
                "inference temperature schedules currently require proposal='dlmc' "
                "without guidance; local transition mixing is supported only for "
                "mixing_scheme='mean_energy'"
            )
        if rejection_step:
            raise NotImplementedError(
                "per-observation inference temperature schedules do not support "
                "a rejection step"
            )
        if temperature_schedule_override is None:
            if use_mean_energy_mixture_schedule:
                assert local_mixture is not None
                _, frozen_local_energy = _compute_mean_energy_local_energy(
                    S,
                    h,
                    J,
                    edge_idx,
                    local_mixture,
                    penalty_func=penalty_func,
                    differentiable_penalty=differentiable_penalty,
                )
            else:
                _, frozen_local_energy = _compute_dlmc_local_energy(
                    S,
                    h,
                    J,
                    edge_idx,
                    penalty_func=penalty_func,
                    differentiable_penalty=differentiable_penalty,
                )
            scheduled_site_mask = (mask_mutatable > 0) & (mask_i > 0)
            temperature_schedule_trace = build_inference_schedule_trace(
                str(temperature_schedule_cfg["name"]),
                local_energy=frozen_local_energy.detach(),
                legal_state_mask=mask_S > 0,
                site_mask=scheduled_site_mask,
                start_temperature=float(temperature_init),
                end_temperature=float(temperature),
                total_steps=int(num_iterations),
                end_hold_steps=int(
                    temperature_schedule_cfg.get("end_hold_steps", 50)
                ),
                variance_floor=float(
                    temperature_schedule_cfg.get("variance_floor", 1e-12)
                ),
                bisection_iterations=int(
                    temperature_schedule_cfg.get("bisection_iterations", 48)
                ),
            )
            temperatures = temperature_schedule_trace.temperatures
        else:
            temperatures = torch.as_tensor(
                temperature_schedule_override,
                device=h.device,
                dtype=torch.float32,
            )
            expected_shape = (num_iterations, h.shape[0])
            if temperatures.shape != expected_shape:
                raise ValueError(
                    "temperature_schedule_override must have shape "
                    f"{expected_shape}, got {tuple(temperatures.shape)}"
                )
            if not bool(torch.isfinite(temperatures).all()) or bool(
                (temperatures <= 0).any()
            ):
                raise ValueError(
                    "temperature_schedule_override must be positive and finite"
                )
    else:
        if return_temperature_schedule:
            raise ValueError(
                "return_temperature_schedule requires an enabled temperature schedule"
            )
        num_iterations_annealing = int(annealing_fraction * num_iterations)
        temperatures = np.linspace(
            temperature_init, temperature, num_iterations_annealing
        ).tolist() + [temperature] * (num_iterations - num_iterations_annealing)

    proposal_lambda_pocket: list[torch.Tensor] = []
    proposal_objective_weights: list[torch.Tensor] = []
    if proposal_mode is PottsProposalMode.LOCAL_MIXTURE_DLMC:
        assert local_mixture is not None

        if isinstance(local_mixture, PottsMultiObjectiveMixture):

            def _energy_proposal(_S, _T, _gamma=None):
                U_local, logp_local, objective_weights = (
                    _potts_proposal_dlmc_multiobjective(
                        _S,
                        h,
                        J,
                        edge_idx,
                        local_mixture,
                        legal_state_mask=legal_state_mask,
                        T=_T,
                        penalty_func=penalty_func,
                        differentiable_penalty=differentiable_penalty,
                        dt=dlmc_dt,
                    )
                )
                return _PottsProposalResult(
                    energy=U_local,
                    log_probs=logp_local,
                    objective_weights=objective_weights,
                )

        else:

            def _energy_proposal(_S, _T, _gamma=None):
                U_local, logp_local, lambda_pocket = (
                    _potts_proposal_dlmc_local_mixture(
                        _S,
                        h,
                        J,
                        edge_idx,
                        local_mixture,
                        legal_state_mask=legal_state_mask,
                        T=_T,
                        penalty_func=penalty_func,
                        differentiable_penalty=differentiable_penalty,
                        dt=dlmc_dt,
                    )
                )
                return _PottsProposalResult(
                    energy=U_local,
                    log_probs=logp_local,
                    lambda_pocket=lambda_pocket,
                )

    elif proposal_mode is PottsProposalMode.GUIDED_DLMC:

        def _energy_proposal(_S, _T, _gamma):
            U_local, logp_local = _potts_proposal_dlmc_guidance_energy(
                _S,
                h,
                J,
                edge_idx,
                h_uncond,
                J_uncond,
                edge_idx_uncond,
                gamma=_gamma,
                T=_T,
                penalty_func=penalty_func,
                differentiable_penalty=differentiable_penalty,
                dt=dlmc_dt,
            )
            return _PottsProposalResult(energy=U_local, log_probs=logp_local)

    elif proposal_mode is PottsProposalMode.CHROMATIC:

        def _energy_proposal(_S, _T, _gamma=None):
            U_local, logp_local = _potts_proposal_gibbs(
                _S,
                h,
                J,
                edge_idx,
                T=_T,
                penalty_func=penalty_func,
                differentiable_penalty=differentiable_penalty,
            )
            return _PottsProposalResult(energy=U_local, log_probs=logp_local)

    elif proposal_mode is PottsProposalMode.DLMC:

        def _energy_proposal(_S, _T, _gamma=None):
            U_local, logp_local = _potts_proposal_dlmc(
                _S,
                h,
                J,
                edge_idx,
                T=_T,
                penalty_func=penalty_func,
                differentiable_penalty=differentiable_penalty,
                dt=dlmc_dt,
            )
            return _PottsProposalResult(energy=U_local, log_probs=logp_local)
    else:
        raise NotImplementedError

    # Per-step γ trajectory. When no schedule is supplied (or schedule is the
    # constant type), this collapses to [gamma] * num_iterations and behavior
    # is bit-for-bit identical to the legacy constant-γ path.
    if (
        use_guidance
        and gamma_schedule_cfg is not None
        and gamma_schedule_cfg.get("type", "constant") != "constant"
    ):
        gammas_per_step = _build_gamma_schedule(num_iterations, gamma_schedule_cfg)
    else:
        gammas_per_step = [float(gamma)] * num_iterations

    cumulative_sweeps = 0
    if return_trajectory:
        S_trajectory = []
        U_trajectory = []
    for i, T_i in enumerate(tqdm(temperatures, desc="Potts Sampling", leave=False)):
        if isinstance(T_i, torch.Tensor):
            T_proposal = T_i[:, None, None]
        else:
            T_proposal = T_i
        g_i = gammas_per_step[i]
        # Cycle through Gibbs updates random sites to the update with fixed prob
        if proposal_mode is PottsProposalMode.CHROMATIC:
            mask_update = schedule.eq(i % num_colors)
        else:
            mask_update = torch.ones_like(S) > 0
        if mask_mutatable is not None:
            mask_update = mask_update * (mask_mutatable > 0)

        # Compute current energy and local conditionals
        proposal_result = _energy_proposal(S, T_proposal, g_i)
        U = proposal_result.energy
        logp = proposal_result.log_probs
        if return_mixing_diagnostics:
            if isinstance(local_mixture, PottsMultiObjectiveMixture):
                assert proposal_result.objective_weights is not None
                proposal_objective_weights.append(
                    proposal_result.objective_weights.detach()
                )
            else:
                assert proposal_result.lambda_pocket is not None
                proposal_lambda_pocket.append(
                    proposal_result.lambda_pocket.detach()
                )

        # Propose
        S_new = torch.distributions.categorical.Categorical(logits=logp).sample()
        S_new = torch.where(mask_update, S_new, S)
        #* As padded positions only have 1 at index 0, they will be always alanine anyway

        # Metropolis-Hastings adjusment
        if rejection_step:

            def _flux(_U, _logp, _S):
                logp_transition = torch.gather(_logp, -1, _S[..., None])
                _logp_ij = (mask_update.float() * logp_transition[..., 0]).sum(1)
                flux = -_U / T_i + _logp_ij
                return flux

            new_proposal_result = _energy_proposal(S_new, T_proposal, g_i)
            U_new = new_proposal_result.energy
            logp_new = new_proposal_result.log_probs

            _flux_backward = _flux(U_new, logp_new, S)
            _flux_forward = _flux(U, logp, S_new)
            acc_ratio = torch.exp((_flux_backward - _flux_forward)).clamp(max=1.0)
            if verbose:  # and i % 100 == 0:
                print(
                    f"{(U_new - U).mean().item():0.2f}"
                    f"\t{(_flux_backward - _flux_forward).mean().item():0.2f}"
                    f"\t{acc_ratio.mean().item():0.2f}"
                )
            u = torch.bernoulli(acc_ratio)[..., None]
            S = torch.where(u > 0, S_new, S)
            cumulative_sweeps += (u * mask_update).sum(1).mean().item() / S.shape[1]
        else:
            S = S_new
            cumulative_sweeps += (mask_update).float().sum(1).mean().item() / S.shape[1]

        if return_trajectory and i % (thin_sweeps) == 0:
            S_trajectory.append(S)
            U_trajectory.append(U)

    terminal_objective_weights = None
    if proposal_mode is PottsProposalMode.LOCAL_MIXTURE_DLMC:
        assert local_mixture is not None
        if isinstance(local_mixture, PottsMultiObjectiveMixture):
            U, terminal_objective_weights = _potts_multiobjective_total_energy(
                S,
                h,
                J,
                edge_idx,
                local_mixture,
                legal_state_mask=legal_state_mask,
                penalty_func=penalty_func,
                differentiable_penalty=differentiable_penalty,
            )
        else:
            U = _potts_local_mixture_total_energy(
                S,
                h,
                J,
                edge_idx,
                local_mixture,
                legal_state_mask=legal_state_mask,
                penalty_func=penalty_func,
                differentiable_penalty=differentiable_penalty,
            )
    elif proposal_mode is PottsProposalMode.GUIDED_DLMC:
        # Keep the reported U consistent with the mixed distribution
        # we are actually sampling from (penalty-free; raw physical Potts
        # energies on both branches, mixed with the terminal γ).
        g_final = gammas_per_step[-1] if gammas_per_step else float(gamma)
        U_cond_final, _ = compute_potts_energy(S, h, J, edge_idx)
        U_uncond_final, _ = compute_potts_energy(
            S, h_uncond, J_uncond, edge_idx_uncond
        )
        U = g_final * U_cond_final + (1.0 - g_final) * U_uncond_final
    else:
        U, _ = compute_potts_energy(S, h, J, edge_idx)

    if verbose:
        print(f"Effective number of sweeps: {cumulative_sweeps}")
    if return_temperature_schedule:
        if return_trajectory:
            raise NotImplementedError(
                "temperature schedule traces cannot be combined with state trajectories"
            )
        assert temperature_schedule_trace is not None
        return S, U, temperature_schedule_trace
    if return_mixing_diagnostics:
        if isinstance(local_mixture, PottsMultiObjectiveMixture):
            if not proposal_objective_weights:
                raise RuntimeError(
                    "multi-objective pcEBM sampling produced no mixing weights"
                )
            assert terminal_objective_weights is not None
            weight_trace = torch.stack(proposal_objective_weights, dim=0)
            objective_labels = (
                local_mixture.primary_label,
                *(
                    objective.label
                    for objective in local_mixture.additional_objectives
                ),
            )
            mixing_diagnostics = PottsMultiObjectiveDiagnostics(
                objective_labels=objective_labels,
                objective_weights_last_proposal=weight_trace[-1],
                objective_weights_terminal=terminal_objective_weights,
                objective_weights_min=weight_trace.min(dim=0).values,
                objective_weights_max=weight_trace.max(dim=0).values,
                objective_weights_mean=weight_trace.mean(dim=0),
            )
            return S, U, mixing_diagnostics
        if not proposal_lambda_pocket:
            raise RuntimeError("pcEBM sampling produced no mixing weights")
        lambda_trace = torch.stack(proposal_lambda_pocket, dim=0)
        mixing_diagnostics = PottsMixingDiagnostics(
            lambda_pocket_final=lambda_trace[-1],
            lambda_pocket_min=lambda_trace.min(dim=0).values,
            lambda_pocket_max=lambda_trace.max(dim=0).values,
            lambda_pocket_mean=lambda_trace.mean(dim=0),
        )
        return S, U, mixing_diagnostics
    if return_trajectory:
        return S, U, S_trajectory, U_trajectory
    else:
        return S, U


def init_sampling_masks(
    logits_init: torch.Tensor,
    mask_sample: Optional[torch.Tensor] = None,
    S: Optional[torch.LongTensor] = None,
    ban_S: Optional[list[int]] = None,
    pos_restrict_aatype: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    """Parse sampling masks and an initial sequence.

    Args:
        logits_init (torch.Tensor): Logits for sequence initialization with shape
            `(num_batch, num_nodes, alphabet)`.
        mask_sample (torch.Tensor, optional): Binary sampling mask indicating which
            positions are free to change with shape `(num_batch, num_nodes)` or which
            tokens are valid at each position with shape
            `(num_batch, num_nodes, alphabet)`. In the latter case, `mask_sample` will
            take priority over `S` except for positions in which `mask_sample` is
            all zero.
        S (torch.LongTensor optional): Initial sequence with shape
            `(num_batch, num_nodes)`.
        ban_S (list of int, optional): Optional list of alphabet indices to ban from
            all positions during sampling.
        pos_restrict_aatype (tuple of torch.Tensor, optional): Tuple of two tensors with shape
            `(num_batch, num_nodes)` and `(num_batch, num_nodes, alphabet)`, respectively,
            indicating which positions are restricted to certain aatypes and which aatypes
            are allowed at each position.

    Returns:
        mask_sample (torch.Tensor): Finalized position specific mask with shape
            `(num_batch, num_nodes, alphabet)`.
        S (torch.Tensor): Self-consistent initial `S` with shape
            `(num_batch, num_nodes)`.
    """

    if S is None and mask_sample is not None:
        raise Exception("To use masked sampling, please provide an initial S")

    if mask_sample is None:
        mask_S = torch.ones_like(logits_init)
    elif mask_sample.dim() == 2:
        # Position-restricted sampling
        # Used for generating initial mask for potts sampling
        # mask_sample: mask for positions that are free to sample
        # mask_sample: (B, N), logits_init: (B, N, const.AF3_ENCODING.n_tokens)
        # S: (B, N). Initial sequence.
        mask_sample_expand = mask_sample[..., None].expand(logits_init.shape) # (B, N, const.AF3_ENCODING.n_tokens)
        # mask_sample is 1 for positions that are free to sample, 0 for positions that are not free to sample
        O_init = F.one_hot(S, logits_init.shape[-1]).float() # (B, N, const.AF3_ENCODING.n_tokens)
        mask_S = mask_sample_expand + (1 - mask_sample_expand) * O_init
        # Since O_init is 0 for padded positions, mask_S[b, padded_positions, 0] = 1
        # mask_S is 0 for non-padded positions, and 1 for padded positions.
    elif mask_sample.dim() == 3:
        O_init = F.one_hot(S, logits_init.shape[-1]).float()
        # Mutation-restricted sampling
        mask_zero = (mask_sample.sum(-1, keepdim=True) == 0).float()
        # for padded_positions, mask_sample[b, padded_positions, 0] = 1
        # So mask_zero is 0 for padded positions
        mask_S = ((mask_zero * O_init + mask_sample) > 0).float()
        # And thus mask_S[b, padded_positions, 0] = 1
    else:
        raise NotImplementedError

    # Handle aatype restrictions
    if ban_S is not None:
        # ban certain aatypes
        mask_S[:, :, ban_S] = 0.0
        # ban_S = {"X"} + const.AF3_ENCODING.encode(const.AF3_ENCODING.non_protein_tokens)
        # (251109) mask_S is 0.0 for all non-protein tokens for now.

    if pos_restrict_aatype is not None:
        # restrict to certain aatypes at certain positions
        restrict_pos_mask, allowed_aatype_mask = pos_restrict_aatype  # (B, N), (B, N, K)
        mask_S[restrict_pos_mask.bool()] = allowed_aatype_mask[restrict_pos_mask.bool()]
        if ban_S is not None:
            mask_S[:, :, ban_S] = 0.0
        empty_restricted_mask = restrict_pos_mask.bool() & (mask_S.sum(-1) == 0)
        if bool(empty_restricted_mask.any()):
            raise ValueError(
                "pos_restrict_aatype leaves no allowed amino-acid states after applying ban_S"
            )

    mask_S_1D = (mask_S.sum(-1) > 1).float()  # check where we can sample
    # For initial mask generation,
    # padded positions are 0, as mask_S.sum(-1) = 1 for padded positions
    # For the second mask generation, also the same

    logits_init_masked = 1000 * mask_S + logits_init
    #! 1000 where we can sample, 0 where we can't (or don't want to) sample
    S_init = torch.distributions.categorical.Categorical(logits=logits_init_masked).sample()
    S = torch.where(mask_S_1D.bool(), S_init, S)  # where we can sample, set S to S_init
    S = torch.where(mask_S.sum(-1) == 1, mask_S.argmax(-1), S)  # where there is only one possible aatype, set S to the aatype
    # This is why [b, padded_positions, 0] = 1 for S.
    return mask_S, mask_S_1D, S


def _potts_proposal_gibbs(
    S, h, J, edge_idx, T=1.0, penalty_func=None, differentiable_penalty=True
):
    U, U_i = compute_potts_energy(S, h, J, edge_idx)

    if penalty_func is not None:
        if differentiable_penalty:
            with torch.enable_grad():
                S_onehot = F.one_hot(S, h.shape[0 - 1]).float()
                S_onehot.requires_grad = True
                U_penalty = penalty_func(S_onehot)
                U_i_adjustment = torch.autograd.grad(U_penalty.sum(), [S_onehot])[
                    0
                ].detach()
                U_penalty = U_penalty.detach()
            U_i = U_i + 0.5 * U_i_adjustment
        else:
            U_penalty = penalty_func(S_onehot)
        U = U + U_penalty

    logp_i = F.log_softmax(-U_i / T, dim=-1)
    return U, logp_i


def _compute_dlmc_local_energy(
    S,
    h,
    J,
    edge_idx,
    penalty_func=None,
    differentiable_penalty=True,
):
    """Compute the exact local energies consumed by the DLMC conditional."""

    U, U_i = compute_potts_energy(S, h, J, edge_idx)
    if penalty_func is not None:
        O = F.one_hot(S, h.shape[0 - 1]).float()
        if differentiable_penalty:
            with torch.enable_grad():
                O.requires_grad = True
                U_penalty = penalty_func(O)
                U_i_adjustment = torch.autograd.grad(U_penalty.sum(), [O])[0].detach()
                U_penalty = U_penalty.detach()
                U_i_adjustment = U_i_adjustment - torch.gather(
                    U_i_adjustment, -1, S[..., None]
                )
                # Base-off the values by subtracting the U_i_adjustment of the current state
            U_i_mutate = U_i - torch.gather(U_i, -1, S[..., None])
            # Base-off, but it's not used anywhere, why?

            U_i = U_i + U_i_adjustment
        else:
            U_penalty = penalty_func(O)
        U = U + U_penalty
    return U, U_i


def _potts_proposal_dlmc(
    S,
    h,
    J,
    edge_idx,
    T=1.0,
    penalty_func=None,
    differentiable_penalty=True,
    dt=0.1,
    autoscale=True,
    balancing_func="sigmoid",
):
    U, U_i = _compute_dlmc_local_energy(
        S,
        h,
        J,
        edge_idx,
        penalty_func=penalty_func,
        differentiable_penalty=differentiable_penalty,
    )

    # Compute local equilibrium distribution
    logP_j = F.log_softmax(-U_i / T, dim=-1)

    return U, _dlmc_transition_log_probs(
        S,
        logP_j,
        dt=dt,
        balancing_func=balancing_func,
    )


def _dlmc_transition_log_probs(
    S: torch.LongTensor,
    logP_j: torch.Tensor,
    *,
    dt: float | torch.Tensor = 0.1,
    balancing_func: str = "sigmoid",
) -> torch.Tensor:
    """Convert local equilibrium log-probabilities into DLMC transitions."""
    if S.shape != logP_j.shape[:-1]:
        raise ValueError("S and logP_j batch/token axes must match")
    if isinstance(dt, torch.Tensor):
        if dt.shape == S.shape:
            dt = dt.unsqueeze(-1)
        elif dt.shape != (*S.shape, 1):
            raise ValueError(
                "tensor dt must have shape [B,N] or [B,N,1], got "
                f"{tuple(dt.shape)}"
            )
        dt = dt.to(device=logP_j.device, dtype=logP_j.dtype)
        if not bool(torch.isfinite(dt).all()) or bool((dt < 0.0).any()):
            raise ValueError("dt must be finite and nonnegative")
    else:
        dt = float(dt)
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("dt must be finite and nonnegative")
    O = F.one_hot(S, logP_j.shape[-1]).float()
    logP_i = torch.gather(logP_j, -1, S[..., None])
    # log probability of the current state
    if balancing_func == "sqrt":
        log_Q_ij = 0.5 * (logP_j - logP_i)
    elif balancing_func == "sigmoid":
        log_Q_ij = F.logsigmoid(logP_j - logP_i)
    else:
        raise NotImplementedError

    rate = torch.exp(log_Q_ij - logP_j)

    # Compute transition probability
    logP_ij = logP_j + (-(-dt * rate).expm1()).log()
    p_flip = ((1.0 - O) * logP_ij.exp()).sum(-1, keepdim=True)

    # DEBUG:
    # flux = ((1. - O) * torch.exp(log_Q_ij)).mean([1,2], keepdim=True)
    # print(f" ->Flux is {flux.item():0.2f}, FlipProb is {p_flip.mean():0.2f}")

    logP_ii = (1.0 - p_flip).clamp(1e-5).log()
    logP_ij = torch.where(O.bool(), logP_ii, logP_ij)
    return logP_ij


def _validate_local_mixture(
    mixture: PottsLocalMixture,
    primary_h: torch.Tensor,
    primary_mask_i: torch.Tensor,
) -> PottsLocalMixture:
    if mixture.mixing_scheme not in POTTS_MIXING_SCHEMES:
        raise ValueError(
            f"mixing_scheme must be one of {sorted(POTTS_MIXING_SCHEMES)}, got "
            f"{mixture.mixing_scheme!r}"
        )
    if mixture.h.shape != primary_h.shape:
        raise ValueError(
            "Pocket/scaffold h shape mismatch: "
            f"{tuple(primary_h.shape)} vs {tuple(mixture.h.shape)}"
        )
    batch_size, num_nodes, num_states = primary_h.shape
    if mixture.J.ndim != 5:
        raise ValueError(
            "Scaffold J must have shape [B,N,K,C,C], got "
            f"{tuple(mixture.J.shape)}"
        )
    if (
        mixture.J.shape[:2] != (batch_size, num_nodes)
        or mixture.J.shape[-2:] != (num_states, num_states)
    ):
        raise ValueError(
            "Pocket/scaffold J graph or alphabet mismatch: "
            f"h={tuple(primary_h.shape)}, scaffold_J={tuple(mixture.J.shape)}"
        )
    if mixture.edge_idx.shape != mixture.J.shape[:3]:
        raise ValueError(
            "Scaffold edge_idx must match J's [B,N,K] axes: "
            f"edge_idx={tuple(mixture.edge_idx.shape)}, J={tuple(mixture.J.shape)}"
        )
    if mixture.mask_i.shape != primary_mask_i.shape:
        raise ValueError(
            "Pocket/scaffold mask_i shape mismatch: "
            f"{tuple(primary_mask_i.shape)} vs {tuple(mixture.mask_i.shape)}"
        )
    if not torch.equal(mixture.mask_i.bool(), primary_mask_i.bool()):
        raise ValueError("Pocket/scaffold protein residue masks do not match")
    alpha = mixture.alpha
    if mixture.mixing_scheme in SPATIAL_POTTS_MIXING_SCHEMES:
        if alpha is None:
            raise ValueError(
                f"alpha is required for {mixture.mixing_scheme!r} mixing"
            )
        if alpha.ndim == primary_h.ndim - 1:
            alpha = alpha.unsqueeze(-1)
        if alpha.shape != primary_h.shape[:-1] + (1,):
            raise ValueError(
                "alpha must have shape [B,N] or [B,N,1] matching h; got "
                f"alpha={tuple(mixture.alpha.shape)}, h={tuple(primary_h.shape)}"
            )
        if not bool(torch.isfinite(alpha).all()):
            raise ValueError("alpha contains non-finite values")
        if bool(((alpha < 0.0) | (alpha > 1.0)).any()):
            raise ValueError("alpha values must be within [0, 1]")
    elif alpha is not None:
        raise ValueError(
            f"alpha must be omitted for {mixture.mixing_scheme!r} mixing"
        )
    return PottsLocalMixture(
        h=mixture.h,
        J=mixture.J,
        edge_idx=mixture.edge_idx,
        mask_i=mixture.mask_i,
        alpha=alpha,
        mixing_scheme=mixture.mixing_scheme,
    )


def _validate_multi_objective_mixture(
    mixture: PottsMultiObjectiveMixture,
    primary_h: torch.Tensor,
    primary_mask_i: torch.Tensor,
) -> PottsMultiObjectiveMixture:
    if mixture.mixing_scheme != "pcebm_multi":
        raise ValueError(
            "multi-objective Potts mixing requires mixing_scheme='pcebm_multi'"
        )
    if not mixture.primary_label:
        raise ValueError("multi-objective pcEBM primary label must be non-empty")
    if not mixture.additional_objectives:
        raise ValueError(
            "multi-objective pcEBM requires at least one additional objective"
        )

    labels = [
        mixture.primary_label,
        *(objective.label for objective in mixture.additional_objectives),
    ]
    if any(not label for label in labels):
        raise ValueError("multi-objective pcEBM labels must be non-empty")
    if len(set(labels)) != len(labels):
        raise ValueError(f"multi-objective pcEBM labels must be unique: {labels}")

    batch_size, num_nodes, num_states = primary_h.shape
    for objective in mixture.additional_objectives:
        if objective.h.shape != primary_h.shape:
            raise ValueError(
                f"Objective {objective.label!r} h shape mismatch: "
                f"{tuple(objective.h.shape)} != {tuple(primary_h.shape)}"
            )
        if objective.J.ndim != 5 or (
            objective.J.shape[:2] != (batch_size, num_nodes)
            or objective.J.shape[-2:] != (num_states, num_states)
        ):
            raise ValueError(
                f"Objective {objective.label!r} J must have compatible "
                f"[B,N,K,C,C] axes, got {tuple(objective.J.shape)}"
            )
        if objective.edge_idx.shape != objective.J.shape[:3]:
            raise ValueError(
                f"Objective {objective.label!r} edge_idx must match J's "
                f"[B,N,K] axes: edge_idx={tuple(objective.edge_idx.shape)}, "
                f"J={tuple(objective.J.shape)}"
            )
        if objective.mask_i.shape != primary_mask_i.shape or not torch.equal(
            objective.mask_i.bool(),
            primary_mask_i.bool(),
        ):
            raise ValueError(
                f"Objective {objective.label!r} protein residue mask does not "
                "match the primary objective"
            )
        if not bool(torch.isfinite(objective.h).all()) or not bool(
            torch.isfinite(objective.J).all()
        ):
            raise ValueError(
                f"Objective {objective.label!r} Potts parameters must be finite"
            )
    return mixture


def _dlmc_penalty_terms(
    S: torch.LongTensor,
    *,
    num_states: int,
    penalty_func: Optional[Callable],
    differentiable_penalty: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if penalty_func is None:
        return None, None
    O = F.one_hot(S, num_states).float()
    if not differentiable_penalty:
        return penalty_func(O), None
    with torch.enable_grad():
        O.requires_grad = True
        U_penalty = penalty_func(O)
        adjustment = torch.autograd.grad(U_penalty.sum(), [O])[0].detach()
        U_penalty = U_penalty.detach()
    adjustment = adjustment - torch.gather(adjustment, -1, S[..., None])
    return U_penalty, adjustment


def _dlmc_penalty_energy(
    S: torch.LongTensor,
    *,
    num_states: int,
    penalty_func: Optional[Callable],
    differentiable_penalty: bool,
) -> torch.Tensor | None:
    """Evaluate only the terminal penalty energy, without its local gradient."""

    if penalty_func is None:
        return None
    O = F.one_hot(S, num_states).float()
    if not differentiable_penalty:
        return penalty_func(O)
    with torch.enable_grad():
        O.requires_grad = True
        return penalty_func(O).detach()


def _mix_local_log_probabilities(
    pocket_U_i: torch.Tensor,
    scaffold_U_i: torch.Tensor,
    alpha: torch.Tensor,
    *,
    mixing_scheme: str,
    temperature: float,
    energy_adjustment: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return local equilibrium log-p_i for probability or energy mixing."""
    if mixing_scheme == "energy":
        mixed_U_i = alpha * pocket_U_i + (1.0 - alpha) * scaffold_U_i
        if energy_adjustment is not None:
            mixed_U_i = mixed_U_i + energy_adjustment
        return F.log_softmax(-mixed_U_i / temperature, dim=-1)
    if mixing_scheme != "probability":
        raise ValueError(f"Unknown local Potts mixing scheme: {mixing_scheme!r}")
    if energy_adjustment is not None:
        pocket_U_i = pocket_U_i + energy_adjustment
        scaffold_U_i = scaffold_U_i + energy_adjustment
    pocket_logp = F.log_softmax(-pocket_U_i / temperature, dim=-1)
    scaffold_logp = F.log_softmax(-scaffold_U_i / temperature, dim=-1)
    neg_inf = torch.full_like(alpha, -torch.inf)
    log_alpha = torch.where(alpha > 0.0, alpha.log(), neg_inf)
    one_minus_alpha = 1.0 - alpha
    log_one_minus_alpha = torch.where(
        one_minus_alpha > 0.0,
        one_minus_alpha.log(),
        neg_inf,
    )
    return torch.logaddexp(
        log_alpha + pocket_logp,
        log_one_minus_alpha + scaffold_logp,
    )


def _simplex_centered_local_energy_gradient(
    local_energy: torch.Tensor,
    legal_state_mask: torch.Tensor,
) -> torch.Tensor:
    """Project local one-hot gradients onto each legal categorical simplex."""
    if legal_state_mask.shape != local_energy.shape:
        raise ValueError("legal state mask must match the local energy field")
    legal_state_mask = legal_state_mask.bool()
    legal_count = legal_state_mask.sum(dim=-1, keepdim=True)
    safe_count = legal_count.clamp_min(1).to(dtype=local_energy.dtype)
    legal_sum = torch.where(
        legal_state_mask,
        local_energy,
        torch.zeros((), dtype=local_energy.dtype, device=local_energy.device),
    ).sum(dim=-1, keepdim=True)
    legal_mean = legal_sum / safe_count
    return torch.where(
        legal_state_mask,
        local_energy - legal_mean,
        torch.zeros((), dtype=local_energy.dtype, device=local_energy.device),
    )


def _pcebm_pocket_weight(
    pocket_gradient: torch.Tensor,
    scaffold_gradient: torch.Tensor,
    active_state_mask: torch.Tensor,
) -> torch.Tensor:
    """Solve the two-objective minimum-norm pcEBM weight per batch item."""
    if pocket_gradient.shape != scaffold_gradient.shape:
        raise ValueError("pcEBM pocket/scaffold gradients must have equal shapes")
    if active_state_mask.shape != pocket_gradient.shape:
        raise ValueError("pcEBM active state mask must match the gradient fields")
    active_state_mask = active_state_mask.bool()
    legal_pocket = pocket_gradient.masked_select(active_state_mask)
    legal_scaffold = scaffold_gradient.masked_select(active_state_mask)
    if not bool(torch.isfinite(legal_pocket).all()) or not bool(
        torch.isfinite(legal_scaffold).all()
    ):
        raise ValueError("pcEBM active centered gradients must be finite")

    work_dtype = torch.promote_types(
        torch.promote_types(pocket_gradient.dtype, scaffold_gradient.dtype),
        torch.float32,
    )
    pocket = torch.where(
        active_state_mask,
        pocket_gradient.to(work_dtype),
        torch.zeros((), dtype=work_dtype, device=pocket_gradient.device),
    )
    scaffold = torch.where(
        active_state_mask,
        scaffold_gradient.to(work_dtype),
        torch.zeros((), dtype=work_dtype, device=scaffold_gradient.device),
    )
    difference = pocket - scaffold
    reduce_dims = tuple(range(1, difference.ndim))
    denominator = difference.square().sum(dim=reduce_dims)
    numerator = -(scaffold * difference).sum(dim=reduce_dims)

    degenerate = denominator == 0.0
    safe_denominator = torch.where(
        degenerate,
        torch.ones_like(denominator),
        denominator,
    )
    lambda_pocket = numerator / safe_denominator
    lambda_pocket = lambda_pocket.clamp(0.0, 1.0)
    lambda_pocket = torch.where(
        degenerate,
        torch.full_like(lambda_pocket, 0.5),
        lambda_pocket,
    )
    return lambda_pocket.to(dtype=pocket_gradient.dtype)


def _pcebm_multiobjective_weights(
    centered_gradients: torch.Tensor,
    active_state_mask: torch.Tensor,
) -> torch.Tensor:
    """Solve the minimum-norm convex combination of two or more gradients.

    The objective count is intentionally small for Potts branch mixing.  We
    enumerate every non-empty simplex face, solve its equality-constrained
    quadratic problem in one batched pseudoinverse, and select the feasible
    minimum.  Enumerating larger faces first gives deterministic maximum-support
    tie breaking for degenerate objectives.
    """

    if centered_gradients.ndim < 3:
        raise ValueError(
            "multi-objective pcEBM gradients must have shape [B,K,...]"
        )
    batch_size, num_objectives = centered_gradients.shape[:2]
    if num_objectives < 2:
        raise ValueError("multi-objective pcEBM requires at least two objectives")
    if active_state_mask.shape != (
        batch_size,
        *centered_gradients.shape[2:],
    ):
        raise ValueError(
            "multi-objective pcEBM active state mask must match each gradient"
        )

    active_state_mask = active_state_mask.bool()
    expanded_mask = active_state_mask[:, None].expand_as(centered_gradients)
    legal_gradients = centered_gradients.masked_select(expanded_mask)
    if not bool(torch.isfinite(legal_gradients).all()):
        raise ValueError(
            "multi-objective pcEBM active centered gradients must be finite"
        )

    work_dtype = torch.float64
    gradients = torch.where(
        expanded_mask,
        centered_gradients.to(work_dtype),
        torch.zeros(
            (),
            dtype=work_dtype,
            device=centered_gradients.device,
        ),
    ).reshape(batch_size, num_objectives, -1)
    gram = torch.einsum("bkd,bld->bkl", gradients, gradients)

    face_members = [
        tuple(bool(bits & (1 << objective)) for objective in range(num_objectives))
        for cardinality in range(num_objectives, 0, -1)
        for bits in range(1, 1 << num_objectives)
        if bits.bit_count() == cardinality
    ]
    faces = torch.tensor(
        face_members,
        dtype=torch.bool,
        device=centered_gradients.device,
    )
    face_values = faces.to(work_dtype)
    face_outer = face_values[:, :, None] * face_values[:, None, :]
    num_faces = faces.shape[0]

    systems = torch.zeros(
        batch_size,
        num_faces,
        num_objectives + 1,
        num_objectives + 1,
        dtype=work_dtype,
        device=centered_gradients.device,
    )
    systems[..., :num_objectives, :num_objectives] = (
        gram[:, None] * face_outer[None]
        + torch.diag_embed((~faces).to(work_dtype))[None]
    )
    systems[..., :num_objectives, num_objectives] = face_values[None]
    systems[..., num_objectives, :num_objectives] = face_values[None]
    rhs = torch.zeros(
        batch_size,
        num_faces,
        num_objectives + 1,
        1,
        dtype=work_dtype,
        device=centered_gradients.device,
    )
    rhs[..., num_objectives, 0] = 1.0

    solutions = torch.linalg.pinv(systems, hermitian=True) @ rhs
    candidates = solutions[..., :num_objectives, 0]
    feasibility_tolerance = 1.0e-7
    valid = (
        torch.isfinite(candidates).all(dim=-1)
        & (candidates >= -feasibility_tolerance).all(dim=-1)
        & (
            (candidates.sum(dim=-1) - 1.0).abs()
            <= feasibility_tolerance
        )
    )
    candidates = candidates.clamp_min(0.0)
    candidates = candidates / candidates.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(work_dtype).tiny
    )
    objectives = torch.einsum(
        "bfk,bkl,bfl->bf",
        candidates,
        gram,
        candidates,
    )
    objectives = torch.where(
        valid,
        objectives,
        torch.full_like(objectives, torch.inf),
    )
    if bool(torch.isinf(objectives).all(dim=1).any()):
        raise RuntimeError(
            "multi-objective pcEBM failed to find a feasible simplex weight"
        )

    minimum = objectives.min(dim=1, keepdim=True).values
    scale = gram.abs().amax(dim=(1, 2), keepdim=False).clamp_min(1.0)
    tie_tolerance = (
        1024.0 * torch.finfo(work_dtype).eps * scale[:, None]
    )
    tied_minima = objectives <= minimum + tie_tolerance
    best_face = tied_minima.to(torch.int64).argmax(dim=1)
    weights = candidates[
        torch.arange(batch_size, device=centered_gradients.device),
        best_face,
    ]
    return weights.to(dtype=centered_gradients.dtype)


def _combine_multiobjective_local_energies(
    local_energies: torch.Tensor,
    *,
    legal_state_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine raw local fields using multi-objective pcEBM weights."""

    if local_energies.ndim != 4:
        raise ValueError(
            "multi-objective local energies must have shape [B,K,N,C]"
        )
    centered_gradients = torch.stack(
        [
            _simplex_centered_local_energy_gradient(
                local_energies[:, objective_index],
                legal_state_mask,
            )
            for objective_index in range(local_energies.shape[1])
        ],
        dim=1,
    )
    weights = _pcebm_multiobjective_weights(
        centered_gradients,
        legal_state_mask,
    )
    mixed_local_energy = torch.einsum(
        "bk,bknc->bnc",
        weights,
        local_energies,
    )
    return mixed_local_energy, weights


def _combine_global_local_energies(
    pocket_U_i: torch.Tensor,
    scaffold_U_i: torch.Tensor,
    *,
    mixing_scheme: str,
    legal_state_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Combine raw local energies, using centered gradients only for pcEBM weights."""

    lambda_pocket = None
    if mixing_scheme == "mean_energy":
        mixed_U_i = 0.5 * (pocket_U_i + scaffold_U_i)
    elif mixing_scheme == "pcebm":
        pocket_gradient = _simplex_centered_local_energy_gradient(
            pocket_U_i,
            legal_state_mask,
        )
        scaffold_gradient = _simplex_centered_local_energy_gradient(
            scaffold_U_i,
            legal_state_mask,
        )
        lambda_pocket = _pcebm_pocket_weight(
            pocket_gradient,
            scaffold_gradient,
            legal_state_mask,
        )
        pocket_weight = lambda_pocket[:, None, None]
        mixed_U_i = (
            pocket_weight * pocket_U_i
            + (1.0 - pocket_weight) * scaffold_U_i
        )
    else:
        raise ValueError(f"Unknown global Potts mixing scheme: {mixing_scheme!r}")
    return mixed_U_i, lambda_pocket


def _mix_global_local_energies(
    pocket_U_i: torch.Tensor,
    scaffold_U_i: torch.Tensor,
    *,
    mixing_scheme: str,
    legal_state_mask: torch.Tensor,
    temperature: float | torch.Tensor,
    energy_adjustment: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build one equilibrium distribution from mean-energy or pcEBM local energy."""

    mixed_U_i, lambda_pocket = _combine_global_local_energies(
        pocket_U_i,
        scaffold_U_i,
        mixing_scheme=mixing_scheme,
        legal_state_mask=legal_state_mask,
    )
    if energy_adjustment is not None:
        mixed_U_i = mixed_U_i + energy_adjustment
    return F.log_softmax(-mixed_U_i / temperature, dim=-1), lambda_pocket


def _compute_mean_energy_local_energy(
    S: torch.LongTensor,
    pocket_h: torch.Tensor,
    pocket_J: torch.Tensor,
    pocket_edge_idx: torch.LongTensor,
    mixture: PottsLocalMixture,
    *,
    penalty_func: Optional[Callable] = None,
    differentiable_penalty: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the scalar and local fields of the fixed 0.5/0.5 composition."""

    if mixture.mixing_scheme != "mean_energy":
        raise ValueError(
            "mean-energy local fields require mixing_scheme='mean_energy', got "
            f"{mixture.mixing_scheme!r}"
        )
    pocket_U, pocket_U_i = compute_potts_energy(
        S, pocket_h, pocket_J, pocket_edge_idx
    )
    scaffold_U, scaffold_U_i = compute_potts_energy(
        S, mixture.h, mixture.J, mixture.edge_idx
    )
    U_penalty, energy_adjustment = _dlmc_penalty_terms(
        S,
        num_states=pocket_h.shape[-1],
        penalty_func=penalty_func,
        differentiable_penalty=differentiable_penalty,
    )
    mixed_U_i = 0.5 * (pocket_U_i + scaffold_U_i)
    if energy_adjustment is not None:
        mixed_U_i = mixed_U_i + energy_adjustment
    U = 0.5 * (pocket_U + scaffold_U)
    if U_penalty is not None:
        U = U + U_penalty
    return U, mixed_U_i


def _local_mixture_total_energy_from_branches(
    S: torch.LongTensor,
    pocket_U: torch.Tensor,
    pocket_U_i: torch.Tensor,
    scaffold_U: torch.Tensor,
    scaffold_U_i: torch.Tensor,
    mixture: PottsLocalMixture,
    lambda_pocket: torch.Tensor | None,
) -> torch.Tensor:
    """Combine already-computed branch energies using the proposal semantics."""

    if mixture.mixing_scheme in SPATIAL_POTTS_MIXING_SCHEMES:
        assert mixture.alpha is not None
        pocket_current = torch.gather(pocket_U_i, -1, S[..., None])[..., 0]
        scaffold_current = torch.gather(scaffold_U_i, -1, S[..., None])[..., 0]
        alpha_per_residue = mixture.alpha[..., 0]
        # Spatial probability/energy mixing does not generally define a global
        # potential, so preserve its existing per-residue diagnostic.
        return (
            alpha_per_residue * pocket_current
            + (1.0 - alpha_per_residue) * scaffold_current
        ).sum(dim=-1)
    if mixture.mixing_scheme == "mean_energy":
        return 0.5 * (pocket_U + scaffold_U)
    if mixture.mixing_scheme == "pcebm":
        assert lambda_pocket is not None
        return lambda_pocket * pocket_U + (1.0 - lambda_pocket) * scaffold_U
    raise ValueError(f"Unknown Potts mixing scheme: {mixture.mixing_scheme!r}")


def _potts_local_mixture_total_energy(
    S: torch.LongTensor,
    pocket_h: torch.Tensor,
    pocket_J: torch.Tensor,
    pocket_edge_idx: torch.LongTensor,
    mixture: PottsLocalMixture,
    *,
    legal_state_mask: torch.Tensor,
    penalty_func: Optional[Callable] = None,
    differentiable_penalty: bool = True,
) -> torch.Tensor:
    """Evaluate terminal mixed energy without constructing another transition."""

    pocket_U, pocket_U_i = compute_potts_energy(
        S, pocket_h, pocket_J, pocket_edge_idx
    )
    scaffold_U, scaffold_U_i = compute_potts_energy(
        S, mixture.h, mixture.J, mixture.edge_idx
    )
    lambda_pocket = None
    if mixture.mixing_scheme == "pcebm":
        _, lambda_pocket = _combine_global_local_energies(
            pocket_U_i,
            scaffold_U_i,
            mixing_scheme="pcebm",
            legal_state_mask=legal_state_mask,
        )
    U = _local_mixture_total_energy_from_branches(
        S,
        pocket_U,
        pocket_U_i,
        scaffold_U,
        scaffold_U_i,
        mixture,
        lambda_pocket,
    )
    U_penalty = _dlmc_penalty_energy(
        S,
        num_states=pocket_h.shape[-1],
        penalty_func=penalty_func,
        differentiable_penalty=differentiable_penalty,
    )
    if U_penalty is not None:
        U = U + U_penalty
    return U


def _compute_multiobjective_branch_energies(
    S: torch.LongTensor,
    primary_h: torch.Tensor,
    primary_J: torch.Tensor,
    primary_edge_idx: torch.LongTensor,
    mixture: PottsMultiObjectiveMixture,
) -> tuple[torch.Tensor, torch.Tensor]:
    branch_outputs = [
        compute_potts_energy(S, primary_h, primary_J, primary_edge_idx),
        *(
            compute_potts_energy(
                S,
                objective.h,
                objective.J,
                objective.edge_idx,
            )
            for objective in mixture.additional_objectives
        ),
    ]
    scalar_energies = torch.stack(
        [branch_energy for branch_energy, _ in branch_outputs],
        dim=1,
    )
    local_energies = torch.stack(
        [local_energy for _, local_energy in branch_outputs],
        dim=1,
    )
    return scalar_energies, local_energies


def _potts_multiobjective_total_energy(
    S: torch.LongTensor,
    primary_h: torch.Tensor,
    primary_J: torch.Tensor,
    primary_edge_idx: torch.LongTensor,
    mixture: PottsMultiObjectiveMixture,
    *,
    legal_state_mask: torch.Tensor,
    penalty_func: Optional[Callable] = None,
    differentiable_penalty: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate terminal multi-objective energy and terminal simplex weights."""

    scalar_energies, local_energies = _compute_multiobjective_branch_energies(
        S,
        primary_h,
        primary_J,
        primary_edge_idx,
        mixture,
    )
    _, weights = _combine_multiobjective_local_energies(
        local_energies,
        legal_state_mask=legal_state_mask,
    )
    U = (weights * scalar_energies).sum(dim=1)
    U_penalty = _dlmc_penalty_energy(
        S,
        num_states=primary_h.shape[-1],
        penalty_func=penalty_func,
        differentiable_penalty=differentiable_penalty,
    )
    if U_penalty is not None:
        U = U + U_penalty
    return U, weights


def _potts_proposal_dlmc_multiobjective(
    S: torch.LongTensor,
    primary_h: torch.Tensor,
    primary_J: torch.Tensor,
    primary_edge_idx: torch.LongTensor,
    mixture: PottsMultiObjectiveMixture,
    *,
    legal_state_mask: torch.Tensor,
    T: float = 1.0,
    penalty_func: Optional[Callable] = None,
    differentiable_penalty: bool = True,
    dt: float = 0.1,
    balancing_func: str = "sigmoid",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scalar_energies, local_energies = _compute_multiobjective_branch_energies(
        S,
        primary_h,
        primary_J,
        primary_edge_idx,
        mixture,
    )
    U_penalty, energy_adjustment = _dlmc_penalty_terms(
        S,
        num_states=primary_h.shape[-1],
        penalty_func=penalty_func,
        differentiable_penalty=differentiable_penalty,
    )
    mixed_local_energy, weights = _combine_multiobjective_local_energies(
        local_energies,
        legal_state_mask=legal_state_mask,
    )
    if energy_adjustment is not None:
        mixed_local_energy = mixed_local_energy + energy_adjustment
    U = (weights * scalar_energies).sum(dim=1)
    if U_penalty is not None:
        U = U + U_penalty
    logP_j = F.log_softmax(-mixed_local_energy / T, dim=-1)
    return (
        U,
        _dlmc_transition_log_probs(
            S,
            logP_j,
            dt=dt,
            balancing_func=balancing_func,
        ),
        weights,
    )


def _potts_proposal_dlmc_local_mixture(
    S: torch.LongTensor,
    pocket_h: torch.Tensor,
    pocket_J: torch.Tensor,
    pocket_edge_idx: torch.LongTensor,
    mixture: PottsLocalMixture,
    *,
    legal_state_mask: torch.Tensor,
    T: float = 1.0,
    penalty_func: Optional[Callable] = None,
    differentiable_penalty: bool = True,
    dt: float = 0.1,
    balancing_func: str = "sigmoid",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if mixture.mixing_scheme == "mean_energy":
        U, mixed_U_i = _compute_mean_energy_local_energy(
            S,
            pocket_h,
            pocket_J,
            pocket_edge_idx,
            mixture,
            penalty_func=penalty_func,
            differentiable_penalty=differentiable_penalty,
        )
        return (
            U,
            _dlmc_transition_log_probs(
                S,
                F.log_softmax(-mixed_U_i / T, dim=-1),
                dt=dt,
                balancing_func=balancing_func,
            ),
            None,
        )

    pocket_U, pocket_U_i = compute_potts_energy(
        S, pocket_h, pocket_J, pocket_edge_idx
    )
    scaffold_U, scaffold_U_i = compute_potts_energy(
        S, mixture.h, mixture.J, mixture.edge_idx
    )
    U_penalty, energy_adjustment = _dlmc_penalty_terms(
        S,
        num_states=pocket_h.shape[-1],
        penalty_func=penalty_func,
        differentiable_penalty=differentiable_penalty,
    )
    lambda_pocket = None
    if mixture.mixing_scheme in SPATIAL_POTTS_MIXING_SCHEMES:
        assert mixture.alpha is not None
        logP_j = _mix_local_log_probabilities(
            pocket_U_i,
            scaffold_U_i,
            mixture.alpha,
            mixing_scheme=mixture.mixing_scheme,
            temperature=T,
            energy_adjustment=energy_adjustment,
        )
    else:
        logP_j, lambda_pocket = _mix_global_local_energies(
            pocket_U_i,
            scaffold_U_i,
            mixing_scheme=mixture.mixing_scheme,
            legal_state_mask=legal_state_mask,
            temperature=T,
            energy_adjustment=energy_adjustment,
        )
    U = _local_mixture_total_energy_from_branches(
        S,
        pocket_U,
        pocket_U_i,
        scaffold_U,
        scaffold_U_i,
        mixture,
        lambda_pocket,
    )
    if U_penalty is not None:
        U = U + U_penalty
    return (
        U,
        _dlmc_transition_log_probs(
            S,
            logP_j,
            dt=dt,
            balancing_func=balancing_func,
        ),
        lambda_pocket,
    )


def _potts_proposal_dlmc_guidance_energy(
    S,
    h_cond,
    J_cond,
    edge_idx_cond,
    h_uncond,
    J_uncond,
    edge_idx_uncond,
    gamma=1.0,
    T=1.0,
    penalty_func=None,
    differentiable_penalty=True,
    dt=0.1,
    balancing_func="sigmoid",
):
    """Energy-space CFG: build h_mix, J_mix once, reuse `_potts_proposal_dlmc`.

    Exploits the linearity of the Potts energy in `(h, J)` at fixed `edge_idx`::

        U(S; γ·h_cond + (1-γ)·h_uncond, γ·J_cond + (1-γ)·J_uncond)
            = γ·U_cond(S) + (1-γ)·U_uncond(S)

    so running the standard DLMC proposal on the linearly-mixed parameters
    samples from the Boltzmann distribution of `U_guided = γ·U_cond + (1-γ)·U_uncond`.
    Requires `edge_idx_cond == edge_idx_uncond`, which holds whenever cond and
    uncond differ only in `atom_cond_mask` (same atom positions ⇒ same kNN
    graph). This is asserted at runtime; any future regression where the graph
    depends on the conditioning will surface immediately.

    Complexity penalties are handled by the inner `_potts_proposal_dlmc` on the
    mixed parameters — since the penalty depends only on `S`, this is equivalent
    to applying it once and sharing it across both branches.
    """
    assert h_cond.shape == h_uncond.shape, (
        f"cond/uncond h shape mismatch: {tuple(h_cond.shape)} vs {tuple(h_uncond.shape)}"
    )
    assert J_cond.shape == J_uncond.shape, (
        f"cond/uncond J shape mismatch: {tuple(J_cond.shape)} vs {tuple(J_uncond.shape)}"
    )
    assert edge_idx_cond.shape == edge_idx_uncond.shape, (
        f"cond/uncond edge_idx shape mismatch: "
        f"{tuple(edge_idx_cond.shape)} vs {tuple(edge_idx_uncond.shape)}"
    )
    assert torch.equal(edge_idx_cond, edge_idx_uncond), (
        "edge_idx mismatch between cond/uncond Potts branches — energy-space "
        "guidance requires the two branches to share the same neighbor graph."
    )

    h_mix = gamma * h_cond + (1.0 - gamma) * h_uncond
    J_mix = gamma * J_cond + (1.0 - gamma) * J_uncond

    return _potts_proposal_dlmc(
        S,
        h_mix,
        J_mix,
        edge_idx_cond,
        T=T,
        penalty_func=penalty_func,
        differentiable_penalty=differentiable_penalty,
        dt=dt,
        balancing_func=balancing_func,
    )


def _mask_J(edge_idx, mask_i, mask_ij):
    # Remove self edges
    device = edge_idx.device
    ii = torch.arange(edge_idx.shape[1]).view((1, -1, 1)).to(device)
    not_self = torch.ne(edge_idx, ii).type(torch.float32)

    # Remove missing edges
    self_present = mask_i.unsqueeze(-1)
    neighbor_present = graph.collect_neighbors(self_present, edge_idx)
    neighbor_present = neighbor_present.squeeze(-1)

    mask_J = not_self * self_present * neighbor_present
    if mask_ij is not None:
        mask_J = mask_ij * mask_J
    return mask_J


def build_coupling_mask(
    edge_idx: torch.LongTensor,
    mask_i: torch.Tensor,
    mask_ij: torch.Tensor | None,
    *,
    require_reciprocal: bool,
) -> torch.Tensor:
    """Return the support of couplings that can be nonzero in the Potts graph."""
    coupling_mask = _mask_J(edge_idx, mask_i, mask_ij)
    if require_reciprocal:
        _, coupling_mask = graph.transpose_edge_idx(edge_idx, coupling_mask)
    return coupling_mask


def pseudolikelihood(
    S: torch.LongTensor,
    h: torch.Tensor,
    J: torch.Tensor,
    edge_idx: torch.LongTensor,
) -> torch.Tensor:
    """Compute Potts pseudolikelihood log-probabilities from a sequence.

    Module-level mirror of :meth:`GraphPotts.pseudolikelihood` so loss code can
    call it without a GraphPotts instance (same pattern as
    ``log_composite_likelihood``).

    Inputs:
        S (torch.LongTensor): Sequence with shape ``(num_batch, num_nodes)``.
        h (torch.Tensor): Potts fields with shape
            ``(num_batch, num_nodes, num_states)``.
        J (torch.Tensor): Potts couplings with shape
            ``(num_batch, num_nodes, num_neighbors, num_states, num_states)``.
        edge_idx (torch.LongTensor): Edge indices with shape
            ``(num_batch, num_nodes, num_neighbors)``.

    Outputs:
        log_probs (torch.Tensor): Per-site conditional log-probabilities with
            shape ``(num_batch, num_nodes, num_states)``.
    """
    num_states = J.shape[-1]
    S_j = graph.collect_neighbors(S.unsqueeze(-1), edge_idx)
    S_j = S_j.unsqueeze(-1).expand(-1, -1, -1, num_states, -1)
    J_ij = torch.gather(J, -1, S_j).squeeze(-1)
    J_i = J_ij.sum(2)
    logits = h + J_i
    return F.log_softmax(-logits, dim=-1)


def log_pseudolikelihood(
    S: torch.LongTensor,
    h: torch.Tensor,
    J: torch.Tensor,
    edge_idx: torch.LongTensor,
    mask_i: torch.Tensor,
    smoothing_alpha: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Potts per-site pseudolikelihood at the true residue.

    Sibling of :func:`log_composite_likelihood`. Returns the masked log-prob
    tensor and site mask so the loss module aggregates identically in both
    cases (no one-hot target arithmetic in the caller).

    Inputs:
        S (torch.LongTensor): Sequence with shape ``(num_batch, num_nodes)``.
        h (torch.Tensor): Potts fields with shape
            ``(num_batch, num_nodes, num_states)``.
        J (torch.Tensor): Potts couplings with shape
            ``(num_batch, num_nodes, num_neighbors, num_states, num_states)``.
        edge_idx (torch.LongTensor): Edge indices with shape
            ``(num_batch, num_nodes, num_neighbors)``.
        mask_i (torch.Tensor): Node mask with shape ``(num_batch, num_nodes)``.
        smoothing_alpha (float): Label smoothing probability on ``(0, 1)``.

    Outputs:
        logp_i (torch.Tensor): ``log p(S_i | S_{N(i)})`` at the true ``S_i``,
            masked by ``mask_i``, with shape ``(num_batch, num_nodes)``.
        mask_i (torch.Tensor): Site mask (returned for symmetry with
            ``log_composite_likelihood``).
    """
    num_states = J.shape[-1]

    # Full per-site conditional log-prob: logp[b,i,q] = log p(S_i = q | S_{N(i)})
    S_j = graph.collect_neighbors(S.unsqueeze(-1), edge_idx)
    S_j = S_j.unsqueeze(-1).expand(-1, -1, -1, num_states, -1)
    J_ij = torch.gather(J, -1, S_j).squeeze(-1)
    J_i = J_ij.sum(2)
    logp = F.log_softmax(-(h + J_i), dim=-1)

    # Score the true residue at each site.
    logp_i = torch.gather(logp, -1, S.unsqueeze(-1)).squeeze(-1)

    # Optional label smoothing — per-site analog of log_composite_likelihood's
    # pair-level scheme. num_bins = num_states (not num_states**2).
    if smoothing_alpha > 0.0:
        prob_no_smooth = 1.0 - smoothing_alpha
        prob_background = (1.0 - prob_no_smooth) / float(num_states - 1)
        # Corrects for double-counting of the foreground bin inside logp.sum(-1).
        p_foreground = prob_no_smooth - prob_background
        logp_i = p_foreground * logp_i + prob_background * logp.sum(-1)

    logp_i = mask_i * logp_i
    return logp_i, mask_i


def log_composite_likelihood(
    S: torch.LongTensor,
    h: torch.Tensor,
    J: torch.Tensor,
    edge_idx: torch.LongTensor,
    mask_i: torch.Tensor,
    mask_ij: torch.Tensor,
    smoothing_alpha: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Potts pairwise composite likelihoods from sequence.

    Inputs:
        S (torch.LongTensor): Sequence with shape `(num_batch, num_nodes)`.
        h (torch.Tensor): Potts model fields :math:`h_i(s_i)` with shape
            `(num_batch, num_nodes, num_states)`.
        J (Tensor): Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
            `(num_batch, num_nodes, num_neighbors, num_states, num_states)`.
        edge_idx (torch.LongTensor): Edge indices with shape
            `(num_batch, num_nodes, num_neighbors)`.
        mask_i (torch.Tensor): Node mask with shape `(num_batch, num_nodes)`
        mask_ij (torch.Tensor): Edge mask with shape
            `(num_batch, num_nodes, num_neighbors)`.
        smoothing_alpha (float): Label smoothing probability on `(0,1)`.

    Outputs:
        logp_ij (torch.Tensor): Potts pairwise composite likelihoods evaluated
            for the current sequence with shape
            `(num_batch, num_nodes, num_neighbors)`.
        mask_p_ij (torch.Tensor): Edge mask with shape
            `(num_batch, num_nodes, num_neighbors)`.
    """
    num_batch, num_residues, num_k, num_states, _ = list(J.size())

    # Gather J clamped at j
    # [Batch,i,j,A_i,A_j] => J_ij(:,A_j) [Batch,i,j,A_i]
    S_j = graph.collect_neighbors(S.unsqueeze(-1), edge_idx)
    S_j = S_j.unsqueeze(-1).expand(-1, -1, -1, num_states, -1)
    # (B,i,j,A_i)
    J_clamp_j = torch.gather(J, -1, S_j).squeeze(-1)

    # Gather J clamped at i
    S_i = S.view(num_batch, num_residues, 1, 1, 1)
    S_i = S_i.expand(-1, -1, num_k, num_states, num_states)
    # (B,i,j,1,A_j)
    J_clamp_i = torch.gather(J, -2, S_i)

    # Compute background per-site contributions that sum out J
    # (B,i,j,A_i) => (B,i,A_i)
    r_i = h + J_clamp_j.sum(2)
    r_j = graph.collect_neighbors(r_i, edge_idx)

    # Remove J_ij from the i contributions
    # (B,i,A_i) => (B,i,:,A_i,:)
    r_i = r_i.view([num_batch, num_residues, 1, num_states, 1])
    r_i_minus_ij = r_i - J_clamp_j.unsqueeze(-1)

    # Remove J_ji from the j contributions
    # (B,j,A_j) => (B,:,j,:,A_j)
    r_j = r_j.view([num_batch, num_residues, num_k, 1, num_states])
    r_j_minus_ji = r_j - J_clamp_i

    # Composite likelihood (B,i,j,A_i,A_j)
    logits_ij = r_i_minus_ij + r_j_minus_ji + J
    logits_ij = logits_ij.view([num_batch, num_residues, num_k, -1])
    logp = F.log_softmax(-logits_ij, dim=-1)
    logp = logp.view([num_batch, num_residues, num_k, num_states, num_states])

    # Score the current sequence under
    # (B,i,j,A_i,A_j) => (B,i,j,A_i) => (B,i,j)
    logp_j = torch.gather(logp, -1, S_j).squeeze(-1)
    S_i = S.view(num_batch, num_residues, 1, 1).expand(-1, -1, num_k, -1)
    logp_ij = torch.gather(logp_j, -1, S_i).squeeze(-1)

    # Optional label smoothing (scaled assuming per-token smoothing )
    if smoothing_alpha > 0.0:
        # Foreground probability
        num_bins = num_states ** 2
        prob_no_smooth = (1.0 - smoothing_alpha) ** 2
        prob_background = (1.0 - prob_no_smooth) / float(num_bins - 1)
        # The second term corrects for double counting in background sum
        p_foreground = prob_no_smooth - prob_background
        logp_ij = p_foreground * logp_ij + prob_background * logp.sum([-2, -1])

    mask_p_ij = _mask_J(edge_idx, mask_i, mask_ij)
    logp_ij = mask_p_ij * logp_ij
    return logp_ij, mask_p_ij
