from omegaconf import DictConfig

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchtyping import TensorType

from allatom_design.data.const import PERIODIC_TABLE_FEATURES
from allatom_design.model.seq_denoiser.denoisers.seq_design.mpnn_utils import (
    gather_edges,
    gather_nodes,
)


def build_f_block_features(atomic_numbers: torch.Tensor) -> torch.Tensor:
    """Return PLACER-style lanthanide/actinide features for atomic numbers."""
    atomic_numbers = atomic_numbers.long()
    is_lanthanide = (atomic_numbers >= 57) & (atomic_numbers <= 71)
    is_actinide = (atomic_numbers >= 89) & (atomic_numbers <= 103)
    lanthanide_group = F.one_hot(
        (atomic_numbers - 57).clamp(min=0, max=14),
        num_classes=15,
    ) * is_lanthanide.unsqueeze(-1)
    actinide_group = F.one_hot(
        (atomic_numbers - 89).clamp(min=0, max=14),
        num_classes=15,
    ) * is_actinide.unsqueeze(-1)
    return torch.cat(
        (
            is_lanthanide.unsqueeze(-1),
            is_actinide.unsqueeze(-1),
            lanthanide_group,
            actinide_group,
        ),
        dim=-1,
    ).float()


def gather_dense_pair_features(
    pair_features: torch.Tensor,
    row_indices: torch.Tensor,
    col_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather a dense batched pair tensor at per-token context indices."""
    if pair_features.dim() not in (3, 4):
        raise ValueError(
            "pair_features must have shape [B, L, L] or [B, L, L, C], "
            f"got {tuple(pair_features.shape)}"
        )
    if row_indices.dim() != 3:
        raise ValueError(
            f"row_indices must have shape [B, N, M], got {tuple(row_indices.shape)}"
        )
    if col_indices is None:
        col_indices = row_indices
    if col_indices.shape != row_indices.shape:
        raise ValueError(
            "row_indices and col_indices must have the same shape, got "
            f"{tuple(row_indices.shape)} and {tuple(col_indices.shape)}"
        )
    if pair_features.shape[0] != row_indices.shape[0]:
        raise ValueError(
            "pair feature and index batch dimensions must match, got "
            f"{pair_features.shape[0]} and {row_indices.shape[0]}"
        )
    if pair_features.shape[1] != pair_features.shape[2]:
        raise ValueError(f"pair_features must be square, got {tuple(pair_features.shape)}")

    batch_indices = torch.arange(
        pair_features.shape[0],
        device=pair_features.device,
    )[:, None, None, None]
    return pair_features[
        batch_indices,
        row_indices[:, :, :, None],
        col_indices[:, :, None, :],
    ]


class TokenFeatures(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        Extract token-level edge features and build KNN graph.
        And also extract ligand-related features if ligand_conditioning is True.
        """
        super().__init__()
        self.cfg = cfg

        # Parameters
        self.masked_distance_fill = cfg.get("masked_distance_fill", 1000.0)
        self.k_neighbors = cfg.k_neighbors
        self.num_positional_embeddings = cfg.num_positional_embeddings

        self.hidden_dim = cfg.hidden_dim
        self.context_pair_hidden_dim = int(
            cfg.get("context_pair_hidden_dim", self.hidden_dim)
        )

        # Positional embeddings
        self.positional_embeddings = PositionalEncodings(self.num_positional_embeddings)

        # RBF-related parameters
        self.num_rbf = cfg.num_rbf
        self.min_rbf_mean = cfg.min_rbf_mean
        self.max_rbf_mean = cfg.max_rbf_mean

        # Protein graph-related parameters
        self.protein_graph_rbf_type = cfg.protein_graph_rbf_type
        self.add_protein_local_frame_orientation = bool(
            cfg.get("add_protein_local_frame_orientation", False)
        )
        num_pairwise_dists = self._num_pairwise_distances_for_rbf_type(self.protein_graph_rbf_type)
        protein_graph_edge_in = self.num_positional_embeddings + self.num_rbf * num_pairwise_dists
        if self.add_protein_local_frame_orientation:
            # Directed source-frame direction (3) plus the flattened relative
            # N-CA-C frame rotation (9).
            protein_graph_edge_in += 12
        self.protein_edge_embedding = nn.Linear(protein_graph_edge_in, self.hidden_dim, bias=False)
        self.norm_protein_edges = nn.LayerNorm(self.hidden_dim)

        self.use_multichain_encoding = cfg.get("use_multichain_encoding", True)
        self.ligand_conditioning = cfg.ligand_conditioning
        self.use_ligand_context = cfg.get("use_ligand_context", True)
        self.ligand_atom_context_num = cfg.get("ligand_atom_context_num", 16)
        self.build_legacy_ligand_interaction = bool(
            cfg.get("build_legacy_ligand_interaction", True)
        )

        # Ligand conditioning-related layers
        if self.ligand_conditioning:
            self.use_ligand_aromatic_atom_feature = cfg.get("use_ligand_aromatic_atom_feature", False)
            self.use_ligand_aromatic_edge_feature = cfg.get("use_ligand_aromatic_edge_feature", False)
            self.use_ligand_f_block_features = cfg.get("use_ligand_f_block_features", False)
            self.use_ligand_asinh_formal_charge = cfg.get("use_ligand_asinh_formal_charge", False)
            self.use_ligand_cached_rdkit_chirality = cfg.get("use_ligand_cached_rdkit_chirality", False)
            self.use_ligand_bond_order = cfg.get("use_ligand_bond_order", False)
            self.use_token_bonds = cfg.get("use_token_bonds", False)
            self.add_hydrogenbond_feature = cfg.get("add_hydrogenbond_feature", False)

            self.ligand_atom_base_feature_dim = 147
            self.ligand_cached_rdkit_chirality_feature_dim = 4
            self.ligand_f_block_feature_dim = 32
            self.ligand_bond_order_feature_dim = 5

            self.protein_ligand_interaction_rbf_type = cfg.get("protein_ligand_interaction_rbf_type", "ncacocb")
            if self.protein_ligand_interaction_rbf_type == "cb":
                num_prot_anchor_atoms = 1
            elif self.protein_ligand_interaction_rbf_type == "ncacocb":
                num_prot_anchor_atoms = 5

            # Linear layer for atom type information embedding
            self.type_linear = (
                torch.nn.Linear(self.ligand_atom_base_feature_dim, 64)
                if self.build_legacy_ligand_interaction
                else None
            )
            self.ligand_f_block_interaction_linear = None
            if (
                self.build_legacy_ligand_interaction
                and self.use_ligand_f_block_features
            ):
                self.ligand_f_block_interaction_linear = torch.nn.Linear(
                    self.ligand_f_block_feature_dim,
                    64,
                    bias=False,
                )
            self.ligand_asinh_formal_charge_interaction_linear = None
            if (
                self.build_legacy_ligand_interaction
                and self.use_ligand_asinh_formal_charge
            ):
                self.ligand_asinh_formal_charge_interaction_linear = torch.nn.Linear(
                    1,
                    64,
                    bias=False,
                )
            self.ligand_cached_rdkit_chirality_v2_interaction_linear = None
            if (
                self.build_legacy_ligand_interaction
                and self.use_ligand_cached_rdkit_chirality
            ):
                self.ligand_cached_rdkit_chirality_v2_interaction_linear = torch.nn.Linear(
                    self.ligand_cached_rdkit_chirality_feature_dim,
                    64,
                    bias=False,
                )
            self.ligand_hydrogenbond_interaction_linear = None
            if (
                self.build_legacy_ligand_interaction
                and self.add_hydrogenbond_feature
            ):
                self.ligand_hydrogenbond_interaction_linear = torch.nn.Linear(
                    3,
                    64,
                    bias=False,
                )

            # Parameters for Ligand-protein interaction layers
            self.add_angle_features = cfg.get("add_angle_features", True)
            num_angle_features = 4 if self.add_angle_features else 0

            self.node_project_down = (
                torch.nn.Linear(
                    self.num_rbf * num_prot_anchor_atoms + 64 + num_angle_features,
                    self.hidden_dim,
                    bias=True,
                )
                if self.build_legacy_ligand_interaction
                else None
            )
            self.token_bond_interaction_linear = None
            if self.build_legacy_ligand_interaction and self.use_token_bonds:
                self.token_bond_interaction_linear = torch.nn.Linear(1, self.hidden_dim, bias=False)
            self.norm_nodes = (
                torch.nn.LayerNorm(self.hidden_dim)
                if self.build_legacy_ligand_interaction
                else None
            )

            # Parameters for Ligand subgraph
            # ligand subgraph nodes
            self.y_nodes = torch.nn.Linear(self.ligand_atom_base_feature_dim, self.hidden_dim, bias=False)
            self.ligand_asinh_formal_charge_linear = None
            if self.use_ligand_asinh_formal_charge:
                self.ligand_asinh_formal_charge_linear = torch.nn.Linear(1, self.hidden_dim, bias=False)
            self.ligand_aromatic_atom_linear = None
            if self.use_ligand_aromatic_atom_feature:
                self.ligand_aromatic_atom_linear = torch.nn.Linear(1, self.hidden_dim, bias=False)
            self.ligand_cached_rdkit_chirality_v2_node_linear = None
            if self.use_ligand_cached_rdkit_chirality:
                self.ligand_cached_rdkit_chirality_v2_node_linear = torch.nn.Linear(
                    self.ligand_cached_rdkit_chirality_feature_dim,
                    self.hidden_dim,
                    bias=False,
                )
            self.ligand_f_block_node_linear = None
            if self.use_ligand_f_block_features:
                self.ligand_f_block_node_linear = torch.nn.Linear(
                    self.ligand_f_block_feature_dim,
                    self.hidden_dim,
                    bias=False,
                )
            self.ligand_hydrogenbond_node_linear = None
            if self.add_hydrogenbond_feature:
                self.ligand_hydrogenbond_node_linear = torch.nn.Linear(
                    3,
                    self.hidden_dim,
                    bias=False,
                )
            self.norm_y_nodes = torch.nn.LayerNorm(self.hidden_dim)

            # ligand subgraph edges
            self.y_edges = torch.nn.Linear(
                self.num_rbf,
                self.context_pair_hidden_dim,
                bias=False,
            )
            self.ligand_aromatic_edge_linear = None
            if self.use_ligand_aromatic_edge_feature:
                self.ligand_aromatic_edge_linear = torch.nn.Linear(
                    1,
                    self.context_pair_hidden_dim,
                    bias=False,
                )
            self.ligand_bond_order_linear = None
            if self.use_ligand_bond_order:
                self.ligand_bond_order_linear = torch.nn.Linear(
                    self.ligand_bond_order_feature_dim,
                    self.context_pair_hidden_dim,
                    bias=False,
                )
            self.token_bond_edge_linear = None
            if self.use_token_bonds:
                self.token_bond_edge_linear = torch.nn.Linear(
                    1,
                    self.context_pair_hidden_dim,
                    bias=False,
                )
            self.norm_y_edges = torch.nn.LayerNorm(self.context_pair_hidden_dim)
    def forward(
        self,
        batch: dict[str, TensorType["b ..."]],
        *,
        return_context_metadata: bool = False,
    ):
        """
        Extract token-level edge features and build KNN graph.
        """
        # calculate n, ca, c, o and pseudo CB coordinates
        X = self._get_protein_token_center_coords(batch) # CA coordinates for protein tokens
        D_neighbors, E_idx = self._dist(X = X, mask = batch["protein_residue_node_mask"])
        E = self._embed_protein_edges(batch=batch, D_neighbors=D_neighbors, E_idx=E_idx)

        context_metadata = None
        if self.ligand_conditioning:
            context_features = self._build_ligand_context_features(
                batch,
                return_metadata=return_context_metadata,
            )
            if return_context_metadata:
                V, Y_nodes, Y_edges, Y_m, context_metadata = context_features
            else:
                V, Y_nodes, Y_edges, Y_m = context_features
        else:
            V = None
            Y_nodes = None
            Y_edges = None
            Y_m = None

        outputs = (E, E_idx, V, Y_nodes, Y_edges, Y_m, D_neighbors)
        if return_context_metadata:
            return (*outputs, context_metadata)
        return outputs

    @staticmethod
    def _num_pairwise_distances_for_rbf_type(rbf_type: str) -> int:
        if rbf_type == "ca":
            return 1
        if rbf_type == "ncaco":
            return 4 * 4
        if rbf_type == "ncacocb":
            return 5 * 5
        raise ValueError(f"Invalid protein_graph_rbf_type: {rbf_type}. Must be 'ca', 'ncaco', or 'ncacocb'.")

    def _embed_protein_edges(self, batch, D_neighbors, E_idx):
        RBF_backbone = self._get_protein_graph_rbf(batch=batch, D_neighbors=D_neighbors, E_idx=E_idx)
        E_positional = self._get_positional_edge_features(batch=batch, E_idx=E_idx)
        edge_features = [E_positional, RBF_backbone]
        if self.add_protein_local_frame_orientation:
            edge_features.append(
                self._get_protein_local_frame_orientation(batch, E_idx)
            )
        E = torch.cat(edge_features, -1)
        E = self.protein_edge_embedding(E)
        return self.norm_protein_edges(E)

    def _get_protein_local_frame_orientation(self, batch, E_idx):
        """Return invariant directed PP orientation features ``[u_ij, Q_ij]``.

        Each residue frame uses ``CA`` as origin, normalized ``CA -> C`` as
        its first axis, the orthogonalized ``CA -> N`` direction as its second
        axis, and their cross product as its third axis.  For edge ``i -> j``
        the returned twelve scalars are

        ``R_i^T normalize(CA_j - CA_i)`` and ``vec(R_i^T R_j)``.
        """
        n_coords = batch["noised_n_coords"]
        ca_coords = batch["noised_ca_coords"]
        c_coords = batch["noised_c_coords"]

        first_raw = c_coords - ca_coords
        first_norm = torch.linalg.vector_norm(
            first_raw,
            dim=-1,
            keepdim=True,
        )
        first = F.normalize(first_raw, dim=-1)

        second_seed = n_coords - ca_coords
        second_raw = second_seed - (
            second_seed * first
        ).sum(dim=-1, keepdim=True) * first
        second_norm = torch.linalg.vector_norm(
            second_raw,
            dim=-1,
            keepdim=True,
        )
        second = F.normalize(second_raw, dim=-1)
        third = torch.cross(first, second, dim=-1)
        frames = torch.stack((first, second, third), dim=-1)

        batch_size, num_residues, _, _ = frames.shape
        neighbor_frames = gather_nodes(
            frames.flatten(start_dim=-2),
            E_idx,
        ).reshape(batch_size, num_residues, E_idx.shape[-1], 3, 3)
        neighbor_ca = gather_nodes(ca_coords, E_idx)

        displacement = neighbor_ca - ca_coords.unsqueeze(-2)
        displacement_norm = torch.linalg.vector_norm(
            displacement,
            dim=-1,
            keepdim=True,
        )
        unit_displacement = F.normalize(displacement, dim=-1)

        # Frames store basis vectors as columns.  Contracting their world
        # coordinate axis therefore applies R_i^T.
        source_direction = torch.einsum(
            "bnca,bnkc->bnka",
            frames,
            unit_displacement,
        )
        relative_rotation = torch.einsum(
            "bnca,bnkcd->bnkad",
            frames,
            neighbor_frames,
        )
        orientation = torch.cat(
            (
                source_direction,
                relative_rotation.flatten(start_dim=-2),
            ),
            dim=-1,
        )

        protein_mask = batch["protein_residue_node_mask"].bool()
        frame_valid = (
            (first_norm.squeeze(-1) > 1e-6)
            & (second_norm.squeeze(-1) > 1e-6)
            & protein_mask
        )
        neighbor_frame_valid = gather_nodes(
            frame_valid.unsqueeze(-1),
            E_idx,
        ).squeeze(-1)
        edge_valid = (
            frame_valid.unsqueeze(-1)
            & neighbor_frame_valid
            & (displacement_norm.squeeze(-1) > 1e-6)
        )
        return orientation * edge_valid.unsqueeze(-1).to(orientation.dtype)

    def _get_protein_graph_rbf(self, batch, D_neighbors, E_idx):
        if self.protein_graph_rbf_type == "ca":
            return self._rbf(D_neighbors)
        return self.get_backbone_pseudocb_rbf(
            batch=batch,
            D_neighbors=D_neighbors,
            E_idx=E_idx,
            rbf_type=self.protein_graph_rbf_type,
        )

    def _get_positional_edge_features(self, batch, E_idx):
        residue_index = batch["residue_index"]
        offset = residue_index[:, :, None] - residue_index[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]

        same_chain = ((batch["asym_id"][:, :, None] - batch["asym_id"][:, None, :]) == 0).long()
        E_chains = gather_edges(same_chain[:, :, :, None], E_idx)[:, :, :, 0]
        return self.positional_embeddings(offset.long(), E_chains)

    def _build_ligand_context_features(self, batch, *, return_metadata=False):
        B, N = batch["token_pad_mask"].shape
        device = batch["coords"].device

        noised_coords = batch["noised_coords"]
        noised_backbone_pseudo_cb_coords = self._get_noised_backbone_pseudocb_coords(batch)
        noised_pseudo_cb_coords = batch["noised_pseudo_cb_coords"]
        protein_residue_node_mask = batch["protein_residue_node_mask"]

        ligand_mask = self._get_ligand_context_mask(batch, protein_residue_node_mask)
        noised_ligand_coords = noised_coords * ligand_mask.unsqueeze(-1)
        ligand_atomic_number = batch["atomic_number"] * ligand_mask
        ligand_atom_features, ligand_aromatic = self._build_ligand_atom_features(batch, ligand_mask)

        if self.use_ligand_context:
            Y, Y_t, Y_m, _, Y_idx = self._get_nearest_ligand_atoms(CB = noised_pseudo_cb_coords,
                                                            mask = protein_residue_node_mask,
                                                            Y = noised_ligand_coords,
                                                            Y_t = ligand_atomic_number,
                                                            Y_m = ligand_mask,
                                                            number_of_ligand_atoms = self.ligand_atom_context_num,
                                                            device = device,
                                                            )
        else:
            Y = torch.zeros(B, N, self.ligand_atom_context_num, 3, device=device)
            Y_t = torch.zeros(B, N, self.ligand_atom_context_num, device=device)
            Y_m = torch.zeros(B, N, self.ligand_atom_context_num, device=device)
            Y_idx = torch.zeros(B, N, self.ligand_atom_context_num, dtype=torch.long, device=device)

        Y_m = Y_m.to(dtype=torch.long)
        Y_atom_features = self._gather_ligand_atom_features(
            ligand_atom_features=ligand_atom_features,
            Y_idx=Y_idx,
            B=B,
            N=N,
            device=device,
        )
        Y_aromatic = self._gather_ligand_aromatic_features(
            ligand_aromatic=ligand_aromatic,
            Y_idx=Y_idx,
            B=B,
            N=N,
            device=device,
        )
        Y_t_features, Y_t_embedded, Y_f_block_features = self._embed_ligand_atom_types(Y_t)
        Y_bond_order = self._build_ligand_bond_order_features(batch, Y_idx, Y_m)
        Y_token_bond_edges, Y_token_bond_interactions = self._build_token_bond_features(
            batch=batch,
            Y_idx=Y_idx,
            Y_m=Y_m,
            protein_residue_node_mask=protein_residue_node_mask,
        )
        V = None
        if self.build_legacy_ligand_interaction:
            V = self._embed_ligand_interaction_features(
                Y=Y,
                Y_t_embedded=Y_t_embedded,
                Y_f_block_features=Y_f_block_features,
                Y_atom_features=Y_atom_features,
                Y_token_bond_interactions=Y_token_bond_interactions,
                noised_backbone_pseudo_cb_coords=noised_backbone_pseudo_cb_coords,
            )
        Y_nodes, Y_edges = self._embed_ligand_subgraph_features(
            Y=Y,
            Y_m=Y_m,
            Y_t_features=Y_t_features,
            Y_f_block_features=Y_f_block_features,
            Y_atom_features=Y_atom_features,
            Y_aromatic=Y_aromatic,
            Y_bond_order=Y_bond_order,
            Y_token_bond_edges=Y_token_bond_edges,
        )
        if not return_metadata:
            return V, Y_nodes, Y_edges, Y_m

        atom_name_chars = batch.get("atom_name_chars")
        if atom_name_chars is None:
            context_atom_name_chars = torch.zeros(
                (*Y_idx.shape, 4),
                dtype=torch.long,
                device=Y_idx.device,
            )
        else:
            context_atom_name_chars = self._gather_context_atom_axis(
                atom_name_chars,
                Y_idx,
            )

        molecule_class_features = []
        for feature_name in (
            "atom_is_protein_chain",
            "atom_is_peptide_chain",
            "atom_is_nucleic_acid_chain",
            "atom_is_metal_chain",
            "atom_is_small_molecule_chain",
        ):
            atom_feature = batch.get(feature_name)
            if atom_feature is None:
                gathered_feature = torch.zeros_like(Y_m, dtype=torch.float32)
            else:
                gathered_feature = self._gather_context_atom_axis(
                    atom_feature,
                    Y_idx,
                ).float()
            molecule_class_features.append(gathered_feature)
        context_molecule_class = torch.stack(molecule_class_features, dim=-1)

        context_parent_token_idx = self._gather_context_atom_axis(
            batch["atom_to_token_map"].long(),
            Y_idx,
        )
        flat_context_token_idx = context_parent_token_idx.reshape(B, -1)
        context_asym_id = torch.gather(
            batch["asym_id"].long(),
            dim=1,
            index=flat_context_token_idx,
        ).reshape_as(context_parent_token_idx)

        atom_bond_order = batch.get("atom_ligand_bond_order")
        if atom_bond_order is None:
            context_bond_exists = torch.zeros(
                (*Y_idx.shape, Y_idx.shape[-1]),
                dtype=torch.float32,
                device=Y_idx.device,
            )
        else:
            gathered_bond_order = gather_dense_pair_features(
                atom_bond_order.long(),
                Y_idx,
            )
            context_bond_exists = (
                (gathered_bond_order >= 1) & (gathered_bond_order <= 5)
            ).float()
        context_pair_mask = Y_m[..., :, None] * Y_m[..., None, :]
        context_bond_exists = context_bond_exists * context_pair_mask

        context_metadata = {
            "context_coords": Y,
            "context_atom_idx": Y_idx,
            "context_atom_name_chars": context_atom_name_chars,
            "context_molecule_class": context_molecule_class,
            "context_parent_token_idx": context_parent_token_idx,
            "context_asym_id": context_asym_id,
            "context_bond_exists": context_bond_exists,
            "token_bond_interactions": Y_token_bond_interactions,
            "backbone_anchor_coords": noised_backbone_pseudo_cb_coords,
        }
        return V, Y_nodes, Y_edges, Y_m, context_metadata

    @staticmethod
    def _gather_context_atom_axis(
        atom_feature: torch.Tensor,
        context_atom_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Gather an atom-axis feature at packed local-context indices."""

        batch_size = context_atom_idx.shape[0]
        flat_idx = context_atom_idx.reshape(batch_size, -1)
        if atom_feature.dim() == 2:
            return torch.gather(atom_feature, 1, flat_idx).reshape_as(
                context_atom_idx
            )
        if atom_feature.dim() == 3:
            feature_dim = atom_feature.shape[-1]
            gathered = torch.gather(
                atom_feature,
                1,
                flat_idx[..., None].expand(-1, -1, feature_dim),
            )
            return gathered.reshape(*context_atom_idx.shape, feature_dim)
        raise ValueError(
            "atom_feature must have shape [B,A] or [B,A,C], got "
            f"{tuple(atom_feature.shape)}"
        )

    def _get_noised_backbone_pseudocb_coords(self, batch):
        return torch.cat(
            (
                batch["noised_n_coords"][:, :, None, :],
                batch["noised_ca_coords"][:, :, None, :],
                batch["noised_c_coords"][:, :, None, :],
                batch["noised_o_coords"][:, :, None, :],
                batch["noised_pseudo_cb_coords"][:, :, None, :],
            ),
            dim=2,
        )

    def _get_ligand_context_mask(self, batch, protein_residue_node_mask):
        atom_is_standard_aa_protein = (
            protein_residue_node_mask.gather(dim=-1, index=batch["atom_to_token_map"])
            * batch["atom_pad_mask"]
            * batch["atom_resolved_mask"]
        )
        atom_is_not_standard_aa_protein = (
            (1 - atom_is_standard_aa_protein)
            * batch["atom_pad_mask"]
            * batch["atom_resolved_mask"]
        )
        sidechain_context_atom_mask = batch.get(
            "sidechain_context_atom_mask",
            torch.zeros_like(batch["atom_resolved_mask"]),
        )
        context_atom_mask = (atom_is_not_standard_aa_protein + sidechain_context_atom_mask).clamp(max=1.0)
        return context_atom_mask * batch["atom_cond_mask"]

    def _build_ligand_atom_features(self, batch, ligand_mask):
        ligand_atom_features = {}
        ligand_aromatic = None
        if self.use_ligand_asinh_formal_charge:
            ligand_formal_charge = self._require_batch_feature(
                batch,
                "atom_formal_charge",
                "use_ligand_asinh_formal_charge",
            ).float()
            ligand_atom_features["asinh_formal_charge"] = (
                ligand_formal_charge * ligand_mask
            ).unsqueeze(-1)
        if self.use_ligand_aromatic_atom_feature or self.use_ligand_aromatic_edge_feature:
            ligand_aromatic = batch["atom_is_aromatic"].float() * ligand_mask
            if self.use_ligand_aromatic_atom_feature:
                ligand_atom_features["aromatic_atom"] = ligand_aromatic.unsqueeze(-1)
        if self.use_ligand_cached_rdkit_chirality:
            ligand_chirality_tag = self._require_batch_feature(
                batch,
                "atom_cached_rdkit_chirality_tag",
                "use_ligand_cached_rdkit_chirality",
            ).long()
            ligand_chirality = F.one_hot(
                ligand_chirality_tag.clamp(min=0, max=2),
                num_classes=3,
            ).float()
            ligand_chirality_mask = self._require_batch_feature(
                batch,
                "atom_cached_rdkit_chirality_mask",
                "use_ligand_cached_rdkit_chirality",
            ).float().clamp(min=0.0, max=1.0)
            ligand_atom_features["cached_rdkit_chirality"] = (
                torch.cat(
                    (ligand_chirality, ligand_chirality_mask.unsqueeze(-1)),
                    dim=-1,
                )
                * ligand_mask.unsqueeze(-1)
            )
        if self.add_hydrogenbond_feature:
            ligand_hba = self._require_batch_feature(
                batch,
                "atom_is_HBA",
                "add_hydrogenbond_feature",
            ).float()
            ligand_hbd = self._require_batch_feature(
                batch,
                "atom_is_HBD",
                "add_hydrogenbond_feature",
            ).float()
            ligand_hydrogenbond_mask = self._require_batch_feature(
                batch,
                "atom_hydrogenbond_feature_mask",
                "add_hydrogenbond_feature",
            ).float().clamp(min=0.0, max=1.0)
            ligand_atom_features["hydrogenbond"] = (
                torch.stack(
                    (ligand_hba, ligand_hbd, ligand_hydrogenbond_mask),
                    dim=-1,
                )
                * ligand_mask.unsqueeze(-1)
            )
        return ligand_atom_features, ligand_aromatic

    def _gather_ligand_atom_features(self, ligand_atom_features, Y_idx, B, N, device):
        gathered_features = {}
        feature_specs = (
            ("asinh_formal_charge", self.ligand_asinh_formal_charge_linear, 1),
            ("aromatic_atom", self.ligand_aromatic_atom_linear, 1),
            (
                "cached_rdkit_chirality",
                self.ligand_cached_rdkit_chirality_v2_node_linear,
                self.ligand_cached_rdkit_chirality_feature_dim,
            ),
            ("hydrogenbond", self.ligand_hydrogenbond_node_linear, 3),
        )
        for feature_name, projection, feature_dim in feature_specs:
            if projection is None:
                continue
            gathered_features[feature_name] = self._gather_ligand_atom_feature(
                atom_feature=ligand_atom_features[feature_name],
                feature_dim=feature_dim,
                nn_idx=Y_idx,
                B=B,
                N=N,
                device=device,
            )
        return gathered_features

    def _gather_ligand_atom_feature(self, atom_feature, feature_dim, nn_idx, B, N, device):
        if self.use_ligand_context:
            return self._gather_nearest_atom_features(
                atom_features=atom_feature,
                nn_idx=nn_idx,
                number_of_ligand_atoms=self.ligand_atom_context_num,
                device=device,
            )
        return torch.zeros(
            B,
            N,
            self.ligand_atom_context_num,
            feature_dim,
            device=device,
        )

    def _gather_ligand_aromatic_features(self, ligand_aromatic, Y_idx, B, N, device):
        if not self.use_ligand_aromatic_edge_feature:
            return None
        if self.use_ligand_context and ligand_aromatic is not None:
            return self._gather_nearest_atom_features(
                atom_features=ligand_aromatic,
                nn_idx=Y_idx,
                number_of_ligand_atoms=self.ligand_atom_context_num,
                device=device,
            ).squeeze(-1)
        return torch.zeros(B, N, self.ligand_atom_context_num, device=device)

    def _embed_ligand_atom_types(self, Y_t):
        # Keep the legacy Z/group/period feature tensor checkpoint-compatible.
        Y_t = Y_t.long()
        Y_t_g = torch.tensor(PERIODIC_TABLE_FEATURES[1], device=Y_t.device)[Y_t]
        Y_t_p = torch.tensor(PERIODIC_TABLE_FEATURES[2], device=Y_t.device)[Y_t]
        Y_t_g_1hot_ = torch.nn.functional.one_hot(Y_t_g, 19)
        Y_t_p_1hot_ = torch.nn.functional.one_hot(Y_t_p, 8)
        Y_t_1hot_ = torch.nn.functional.one_hot(Y_t, 120)
        Y_t_features = torch.cat([Y_t_1hot_, Y_t_g_1hot_, Y_t_p_1hot_], -1).float()
        Y_t_embedded = (
            self.type_linear(Y_t_features)
            if self.type_linear is not None
            else None
        )
        Y_f_block_features = build_f_block_features(Y_t)
        return Y_t_features, Y_t_embedded, Y_f_block_features

    def _embed_ligand_interaction_features(
        self,
        Y,
        Y_t_embedded,
        Y_f_block_features,
        Y_atom_features,
        Y_token_bond_interactions,
        noised_backbone_pseudo_cb_coords,
    ):
        if self.protein_ligand_interaction_rbf_type == "cb":
            protein_anchor_coords = noised_backbone_pseudo_cb_coords[:, :, 4:, :]
        else:
            protein_anchor_coords = noised_backbone_pseudo_cb_coords
        D_ligand_to_backbone_or_pseudocb = torch.sqrt(
            torch.sum((Y[:, :, :, None, :] - protein_anchor_coords[:, :, None, :, :]) ** 2, dim=-1) + 1e-6
        )
        RBF_ligand_to_backbone_or_pseudocb = self.compute_rbf_embedding_from_distances(
            D=D_ligand_to_backbone_or_pseudocb
        )
        RBF_ligand_to_backbone_or_pseudocb = RBF_ligand_to_backbone_or_pseudocb.view(
            RBF_ligand_to_backbone_or_pseudocb.shape[0],
            RBF_ligand_to_backbone_or_pseudocb.shape[1],
            RBF_ligand_to_backbone_or_pseudocb.shape[2],
            -1,
        )

        if self.ligand_f_block_interaction_linear is not None:
            Y_t_embedded = Y_t_embedded + self.ligand_f_block_interaction_linear(
                Y_f_block_features
            )
        if self.ligand_asinh_formal_charge_interaction_linear is not None:
            Y_t_embedded = Y_t_embedded + self.ligand_asinh_formal_charge_interaction_linear(
                torch.asinh(Y_atom_features["asinh_formal_charge"].float())
            )
        if self.ligand_cached_rdkit_chirality_v2_interaction_linear is not None:
            chirality_features = Y_atom_features["cached_rdkit_chirality"].float()
            Y_t_embedded = Y_t_embedded + (
                self.ligand_cached_rdkit_chirality_v2_interaction_linear(
                    chirality_features
                )
                * chirality_features[..., -1:]
            )
        if self.ligand_hydrogenbond_interaction_linear is not None:
            hydrogenbond_features = Y_atom_features["hydrogenbond"].float()
            Y_t_embedded = Y_t_embedded + (
                self.ligand_hydrogenbond_interaction_linear(
                    hydrogenbond_features
                )
                * hydrogenbond_features[..., -1:]
            )
        interaction_features = [RBF_ligand_to_backbone_or_pseudocb, Y_t_embedded]
        if self.add_angle_features:
            angle_features = self._make_angle_features(
                noised_backbone_pseudo_cb_coords[:, :, 0, :],
                noised_backbone_pseudo_cb_coords[:, :, 1, :],
                noised_backbone_pseudo_cb_coords[:, :, 2, :],
                Y,
            )
            interaction_features.append(angle_features)
        V = self.node_project_down(torch.cat(interaction_features, dim=-1))
        if self.token_bond_interaction_linear is not None:
            V = V + self.token_bond_interaction_linear(Y_token_bond_interactions)
        return self.norm_nodes(V)

    def _embed_ligand_subgraph_features(
        self,
        Y,
        Y_m,
        Y_t_features,
        Y_f_block_features,
        Y_atom_features,
        Y_aromatic,
        Y_bond_order,
        Y_token_bond_edges,
    ):
        Y_nodes = self.y_nodes(Y_t_features)
        if self.ligand_f_block_node_linear is not None:
            Y_nodes = Y_nodes + self.ligand_f_block_node_linear(Y_f_block_features)
        if self.ligand_asinh_formal_charge_linear is not None:
            Y_nodes = Y_nodes + self.ligand_asinh_formal_charge_linear(
                torch.asinh(Y_atom_features["asinh_formal_charge"].float())
            )
        if self.ligand_aromatic_atom_linear is not None:
            Y_nodes = Y_nodes + self.ligand_aromatic_atom_linear(Y_atom_features["aromatic_atom"].float())
        if self.ligand_cached_rdkit_chirality_v2_node_linear is not None:
            chirality_features = Y_atom_features["cached_rdkit_chirality"].float()
            Y_nodes = Y_nodes + (
                self.ligand_cached_rdkit_chirality_v2_node_linear(
                    chirality_features
                )
                * chirality_features[..., -1:]
            )
        if self.ligand_hydrogenbond_node_linear is not None:
            hydrogenbond_features = Y_atom_features["hydrogenbond"].float()
            Y_nodes = Y_nodes + (
                self.ligand_hydrogenbond_node_linear(hydrogenbond_features)
                * hydrogenbond_features[..., -1:]
            )
        Y_nodes = self.norm_y_nodes(Y_nodes)

        Y_edges = self._rbf(
            torch.sqrt(
                torch.sum((Y[:, :, :, None, :] - Y[:, :, None, :, :]) ** 2, -1) + 1e-6
            )
        )
        Y_edges = self.y_edges(Y_edges)
        if self.ligand_aromatic_edge_linear is not None:
            aromatic_edges = (Y_aromatic[:, :, :, None] * Y_aromatic[:, :, None, :]).unsqueeze(-1)
            y_edge_mask = (Y_m[:, :, :, None] * Y_m[:, :, None, :]).to(dtype=aromatic_edges.dtype).unsqueeze(-1)
            Y_edges = Y_edges + self.ligand_aromatic_edge_linear(aromatic_edges * y_edge_mask)
        if self.ligand_bond_order_linear is not None:
            Y_edges = Y_edges + self.ligand_bond_order_linear(Y_bond_order)
        if self.token_bond_edge_linear is not None:
            Y_edges = Y_edges + self.token_bond_edge_linear(Y_token_bond_edges)
        Y_edges = self.norm_y_edges(Y_edges)
        return Y_nodes, Y_edges

    def _build_ligand_bond_order_features(self, batch, Y_idx, Y_m):
        if not self.use_ligand_bond_order:
            return None
        atom_bond_order = self._require_batch_feature(
            batch,
            "atom_ligand_bond_order",
            "use_ligand_bond_order",
        ).long()
        gathered_bond_order = gather_dense_pair_features(atom_bond_order, Y_idx)
        bonded = (gathered_bond_order >= 1) & (gathered_bond_order <= 5)
        bond_order_features = F.one_hot(
            (gathered_bond_order - 1).clamp(min=0, max=4),
            num_classes=self.ligand_bond_order_feature_dim,
        ).float()
        bond_order_features = bond_order_features * bonded.unsqueeze(-1)

        eligible_atom_mask = torch.zeros_like(batch["atom_pad_mask"], dtype=torch.bool)
        for key in (
            "atom_is_small_molecule_chain",
            "atom_is_metal_chain",
            "atom_is_nucleic_acid_chain",
        ):
            eligible_atom_mask |= self._require_batch_feature(
                batch,
                key,
                "use_ligand_bond_order",
            ).bool()
        gathered_eligibility = self._gather_nearest_atom_features(
            atom_features=eligible_atom_mask,
            nn_idx=Y_idx,
            number_of_ligand_atoms=self.ligand_atom_context_num,
            device=Y_idx.device,
        ).squeeze(-1)
        pair_mask = (
            gathered_eligibility[:, :, :, None]
            & gathered_eligibility[:, :, None, :]
            & Y_m[:, :, :, None].bool()
            & Y_m[:, :, None, :].bool()
        )
        return bond_order_features * pair_mask.unsqueeze(-1)

    def _build_token_bond_features(
        self,
        batch,
        Y_idx,
        Y_m,
        protein_residue_node_mask,
    ):
        if not self.use_token_bonds:
            return None, None
        token_bonds = self._require_batch_feature(
            batch,
            "token_bonds",
            "use_token_bonds",
        ).float()
        atom_to_token_map = batch["atom_to_token_map"].long()
        context_token_ids = torch.gather(
            atom_to_token_map[:, None, :].expand(-1, Y_idx.shape[1], -1),
            dim=2,
            index=Y_idx,
        )

        context_context_bonds = gather_dense_pair_features(
            token_bonds,
            context_token_ids,
        )
        context_pair_mask = (
            Y_m[:, :, :, None].bool() & Y_m[:, :, None, :].bool()
        )
        context_context_bonds = (
            context_context_bonds * context_pair_mask.to(context_context_bonds.dtype)
        ).unsqueeze(-1)

        batch_indices = torch.arange(
            token_bonds.shape[0],
            device=token_bonds.device,
        )[:, None, None]
        current_token_ids = torch.arange(
            Y_idx.shape[1],
            device=token_bonds.device,
        )[None, :, None]
        interaction_bonds = token_bonds[
            batch_indices,
            current_token_ids,
            context_token_ids,
        ]
        interaction_mask = (
            protein_residue_node_mask[:, :, None].bool() & Y_m.bool()
        )
        interaction_bonds = (
            interaction_bonds * interaction_mask.to(interaction_bonds.dtype)
        ).unsqueeze(-1)
        return context_context_bonds, interaction_bonds

    @staticmethod
    def _require_batch_feature(batch, key, feature_flag):
        if key not in batch:
            raise KeyError(
                f"{feature_flag}=true requires batch feature {key!r}"
            )
        return batch[key]


    def _get_protein_token_center_coords(self, batch: dict[str, TensorType["b ..."]]) -> TensorType["b n 3", float]:
        """
        Get protein token-level center coordinates. Standard amino acid only.
        """
        B, N, _ = batch["noised_coords"].shape
        X = batch["noised_coords"][torch.arange(B).unsqueeze(-1), batch["token_to_center_atom"]]  # get center atom for each token, ca for proteins
        X = X * batch["protein_residue_node_mask"].unsqueeze(-1)
        return X

    def _get_token_coords(self, batch: dict[str, TensorType["b ..."]], protein_only: bool = True) -> TensorType["b n 3", float]:
        """
        Get token-level coordinates as an average over all known, resolved atoms in the token.
        """
        B, N, _ = batch["coords"].shape
        X = batch["coords"][torch.arange(B).unsqueeze(-1), batch["token_to_center_atom"]]  # get center atom for each token
        if protein_only:
            X = X * batch["token_is_protein_chain"].unsqueeze(-1)
        X = X * batch["token_exists_mask"].unsqueeze(-1)  # mask out padding and unresolved atoms
        return X

    def _dist(self, X = None, mask = None, eps=1E-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.k_neighbors, X.shape[1]), dim=-1, sorted=True, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = self.min_rbf_mean, self.max_rbf_mean, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def get_backbone_pseudocb_rbf(self, batch: dict[str, TensorType["b ..."]] = None,
                            D_neighbors = None,
                            E_idx = None,
                            rbf_type = "ncacocb") -> TensorType["b n_tokens n_tokens num_rbf", float]:

        ca_coords = batch["noised_ca_coords"]
        n_coords = batch["noised_n_coords"]
        c_coords = batch["noised_c_coords"]
        o_coords = batch["noised_o_coords"]
        pseudo_cb_coords = batch["noised_pseudo_cb_coords"]

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(n_coords, n_coords, E_idx))  # N-N
        RBF_all.append(self._get_rbf(c_coords, c_coords, E_idx))  # C-C
        RBF_all.append(self._get_rbf(o_coords, o_coords, E_idx))  # O-O
        if rbf_type == "ncacocb":
            RBF_all.append(self._get_rbf(pseudo_cb_coords, pseudo_cb_coords, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(ca_coords, n_coords, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(ca_coords, c_coords, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(ca_coords, o_coords, E_idx))  # Ca-O
        if rbf_type == "ncacocb":
            RBF_all.append(self._get_rbf(ca_coords, pseudo_cb_coords, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(n_coords, c_coords, E_idx))  # N-C
        RBF_all.append(self._get_rbf(n_coords, o_coords, E_idx))  # N-O
        if rbf_type == "ncacocb":
            RBF_all.append(self._get_rbf(n_coords, pseudo_cb_coords, E_idx))  # N-Cb
            RBF_all.append(self._get_rbf(pseudo_cb_coords, c_coords, E_idx))  # Cb-C
            RBF_all.append(self._get_rbf(pseudo_cb_coords, o_coords, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(o_coords, c_coords, E_idx))  # O-C
        RBF_all.append(self._get_rbf(n_coords, ca_coords, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(c_coords, ca_coords, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(o_coords, ca_coords, E_idx))  # O-Ca
        if rbf_type == "ncacocb":
            RBF_all.append(self._get_rbf(pseudo_cb_coords, ca_coords, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(c_coords, n_coords, E_idx))  # C-N
        RBF_all.append(self._get_rbf(o_coords, n_coords, E_idx))  # O-N
        if rbf_type == "ncacocb":
            RBF_all.append(self._get_rbf(pseudo_cb_coords, n_coords, E_idx))  # Cb-N
            RBF_all.append(self._get_rbf(c_coords, pseudo_cb_coords, E_idx))  # C-Cb
            RBF_all.append(self._get_rbf(o_coords, pseudo_cb_coords, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(c_coords, o_coords, E_idx))  # C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        return RBF_all

    def compute_rbf_embedding_from_distances(self, D = None):
        """
        Given a tensor of pairwise distances, compute the radial basis
        embedding of the distances.

        Args:
            D (torch.Tensor): [B, L, M] or [B, L, M, N] - Pairwise distances between each
                residue's representative atom, masked by the 2D mask.
        Returns:
            rbf_embedding (torch.Tensor): [B, L, M, num_rbf] or [B, L, M, N, num_rbf] - Radial basis
                function embedding of the pairwise distances.
        """
        # Linear space the means of the radial basis functions.

        rbf_mus = torch.linspace(
            self.min_rbf_mean, self.max_rbf_mean, self.num_rbf, device=D.device
        )

        if len(D.shape) == 3:
            rbf_mus = rbf_mus[None, None, None, :]
        elif len(D.shape) == 4:
            rbf_mus = rbf_mus[None, None, None, None, :]

        # The standard deviation of the radial basis functions.
        rbf_sigma = (self.max_rbf_mean - self.min_rbf_mean) / self.num_rbf

        # Expand the dimensions of D to match the shape of rbf_mus.
        # D_expand: [B, L, M, 1] or [B, L, M, N, 1]
        D_expand = torch.unsqueeze(D, -1)

        # Compute the radial basis function embedding.
        # RBF: [B, L, M, num_rbf] or [B, L, M, N, num_rbf]
        rbf_embedding = torch.exp(-(((D_expand - rbf_mus) / rbf_sigma) ** 2))

        return rbf_embedding

    def _make_angle_features(self, A, B, C, Y): #! from ligandMPNN
        v1 = A - B
        v2 = C - B
        e1 = torch.nn.functional.normalize(v1, dim=-1)
        e1_v2_dot = torch.einsum("bli, bli -> bl", e1, v2)[..., None]
        u2 = v2 - e1 * e1_v2_dot
        e2 = torch.nn.functional.normalize(u2, dim=-1)
        e3 = torch.cross(e1, e2, dim=-1)
        R_residue = torch.cat(
            (e1[:, :, :, None], e2[:, :, :, None], e3[:, :, :, None]), dim=-1
        )

        local_vectors = torch.einsum(
            "blqp, blyq -> blyp", R_residue, Y - B[:, :, None, :]
        )

        rxy = torch.sqrt(local_vectors[..., 0] ** 2 + local_vectors[..., 1] ** 2 + 1e-8)
        f1 = local_vectors[..., 0] / rxy
        f2 = local_vectors[..., 1] / rxy
        rxyz = torch.norm(local_vectors, dim=-1) + 1e-8
        f3 = rxy / rxyz
        f4 = local_vectors[..., 2] / rxyz

        f = torch.cat([f1[..., None], f2[..., None], f3[..., None], f4[..., None]], -1)
        return f

    def _get_nearest_ligand_atoms(self, CB = None,
                                  mask = None,
                                  Y = None,
                                  Y_t = None,
                                  Y_m = None,
                                  number_of_ligand_atoms = 16,
                                  device = None):

        """
        batchfied version of _get_nearest_neighbours in data_utils.py of LigandMPNN.
        """

        mask_CBY = mask[:, :, None] * Y_m[:, None, :]  # [A,B]
        L2_AB = torch.sum((CB[:, :, None, :] - Y[:, None, :, :]) ** 2, -1)
        L2_AB = L2_AB * mask_CBY + (1 - mask_CBY) * self.masked_distance_fill

        nn_idx = torch.argsort(L2_AB, -1)[:, :, :number_of_ligand_atoms]
        L2_AB_nn = torch.gather(L2_AB, -1, nn_idx)
        D_AB_closest = torch.sqrt(L2_AB_nn[:, :, 0])

        # Gather directly from the atom axis.  Repeating the full atom tensor
        # over every protein residue creates an avoidable [B,N,A,...] copy.
        batch_idx = torch.arange(Y.shape[0], device=Y.device)[:, None, None]
        Y_tmp = Y[batch_idx, nn_idx]
        Y_t_tmp = Y_t[batch_idx, nn_idx]
        Y_m_tmp = Y_m[batch_idx, nn_idx] * mask[:, :, None]

        Y = torch.zeros(
            [CB.shape[0], CB.shape[1], number_of_ligand_atoms, 3], dtype=torch.float32, device=device
        )
        Y_t = torch.zeros(
            [CB.shape[0], CB.shape[1], number_of_ligand_atoms], dtype=torch.int32, device=device
        )
        Y_m = torch.zeros(
            [CB.shape[0], CB.shape[1], number_of_ligand_atoms], dtype=torch.int32, device=device
        )
        nn_idx_padded = torch.zeros(
            [CB.shape[0], CB.shape[1], number_of_ligand_atoms], dtype=torch.long, device=device
        )

        num_nn_update = Y_tmp.shape[2]
        Y[:, :, :num_nn_update] = Y_tmp
        Y_t[:, :, :num_nn_update] = Y_t_tmp
        Y_m[:, :, :num_nn_update] = Y_m_tmp
        nn_idx_padded[:, :, :num_nn_update] = nn_idx

        return Y, Y_t, Y_m, D_AB_closest, nn_idx_padded

    def _gather_nearest_atom_features(
        self,
        atom_features: TensorType["b a ..."],
        nn_idx: TensorType["b n m", torch.long],
        number_of_ligand_atoms: int,
        device: torch.device,
    ) -> TensorType["b n m ..."]:
        if atom_features.dim() == 2:
            atom_features = atom_features.unsqueeze(-1)
        feature_dim = atom_features.shape[-1]
        batch_idx = torch.arange(
            atom_features.shape[0],
            device=atom_features.device,
        )[:, None, None]
        gathered_tmp = atom_features[batch_idx, nn_idx]
        gathered = torch.zeros(
            [nn_idx.shape[0], nn_idx.shape[1], number_of_ligand_atoms, feature_dim],
            dtype=atom_features.dtype,
            device=device,
        )
        num_nn_update = gathered_tmp.shape[2]
        gathered[:, :, :num_nn_update] = gathered_tmp
        return gathered

class PositionalEncodings(torch.nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = torch.nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E
