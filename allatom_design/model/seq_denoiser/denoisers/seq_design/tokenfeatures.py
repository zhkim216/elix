from omegaconf import DictConfig

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchtyping import TensorType

from allatom_design.data.const import PERIODIC_TABLE_FEATURES
from allatom_design.model.seq_denoiser.denoisers.seq_design.mpnn_utils import (
    gather_edges,
)


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

        # Positional embeddings
        self.positional_embeddings = PositionalEncodings(self.num_positional_embeddings)

        # RBF-related parameters
        self.num_rbf = cfg.num_rbf
        self.min_rbf_mean = cfg.min_rbf_mean
        self.max_rbf_mean = cfg.max_rbf_mean

        # Protein graph-related parameters
        self.protein_graph_rbf_type = cfg.protein_graph_rbf_type
        num_pairwise_dists = self._num_pairwise_distances_for_rbf_type(self.protein_graph_rbf_type)
        protein_graph_edge_in = self.num_positional_embeddings + self.num_rbf * num_pairwise_dists
        self.protein_edge_embedding = nn.Linear(protein_graph_edge_in, self.hidden_dim, bias=False)
        self.norm_protein_edges = nn.LayerNorm(self.hidden_dim)

        self.use_multichain_encoding = cfg.get("use_multichain_encoding", True)
        self.ligand_conditioning = cfg.ligand_conditioning
        self.use_ligand_context = cfg.get("use_ligand_context", True)
        self.ligand_atom_context_num = cfg.get("ligand_atom_context_num", 16)

        # Ligand conditioning-related layers
        if self.ligand_conditioning:
            self.use_ligand_formal_charge = cfg.get("use_ligand_formal_charge", False)
            self.use_ligand_aromatic_atom_feature = cfg.get("use_ligand_aromatic_atom_feature", False)
            self.use_ligand_aromatic_edge_feature = cfg.get("use_ligand_aromatic_edge_feature", False)
            self.use_ligand_chirality_tag = cfg.get("use_ligand_chirality_tag", False)

            self.ligand_atom_base_feature_dim = 147
            self.ligand_chirality_feature_dim = 3

            self.protein_ligand_interaction_rbf_type = cfg.get("protein_ligand_interaction_rbf_type", "ncacocb")
            if self.protein_ligand_interaction_rbf_type == "cb":
                num_prot_anchor_atoms = 1
            elif self.protein_ligand_interaction_rbf_type == "ncacocb":
                num_prot_anchor_atoms = 5

            # Linear layer for atom type information embedding
            self.type_linear = torch.nn.Linear(self.ligand_atom_base_feature_dim, 64)

            # Parameters for Ligand-protein interaction layers
            self.add_angle_features = cfg.get("add_angle_features", True)
            num_angle_features = 4 if self.add_angle_features else 0

            self.node_project_down = torch.nn.Linear(
                self.num_rbf * num_prot_anchor_atoms + 64 + num_angle_features,
                self.hidden_dim,
                bias=True,
            )
            self.norm_nodes = torch.nn.LayerNorm(self.hidden_dim)

            # Parameters for Ligand subgraph
            # ligand subgraph nodes
            self.y_nodes = torch.nn.Linear(self.ligand_atom_base_feature_dim, self.hidden_dim, bias=False)
            self.ligand_formal_charge_linear = None
            if self.use_ligand_formal_charge:
                self.ligand_formal_charge_linear = torch.nn.Linear(1, self.hidden_dim, bias=False)
            self.ligand_aromatic_atom_linear = None
            if self.use_ligand_aromatic_atom_feature:
                self.ligand_aromatic_atom_linear = torch.nn.Linear(1, self.hidden_dim, bias=False)
            self.ligand_chirality_tag_linear = None
            if self.use_ligand_chirality_tag:
                self.ligand_chirality_tag_linear = torch.nn.Linear(
                    self.ligand_chirality_feature_dim,
                    self.hidden_dim,
                    bias=False,
                )
            self.norm_y_nodes = torch.nn.LayerNorm(self.hidden_dim)

            # ligand subgraph edges
            self.y_edges = torch.nn.Linear(self.num_rbf, self.hidden_dim, bias=False)
            self.ligand_aromatic_edge_linear = None
            if self.use_ligand_aromatic_edge_feature:
                self.ligand_aromatic_edge_linear = torch.nn.Linear(1, self.hidden_dim, bias=False)
            self.norm_y_edges = torch.nn.LayerNorm(self.hidden_dim)


    def zero_init_ligand_feature_projections(self):
        if not self.ligand_conditioning:
            return
        projections = (
            self.ligand_formal_charge_linear,
            self.ligand_aromatic_atom_linear,
            self.ligand_chirality_tag_linear,
            self.ligand_aromatic_edge_linear,
        )
        for projection in projections:
            if projection is not None:
                nn.init.zeros_(projection.weight)
                if projection.bias is not None:
                    nn.init.zeros_(projection.bias)


    def forward(self, batch: dict[str, TensorType["b ..."]]):
        """
        Extract token-level edge features and build KNN graph.
        """
        # calculate n, ca, c, o and pseudo CB coordinates
        X = self._get_protein_token_center_coords(batch) # CA coordinates for protein tokens
        D_neighbors, E_idx = self._dist(X = X, mask = batch["protein_residue_node_mask"])
        E = self._embed_protein_edges(batch=batch, D_neighbors=D_neighbors, E_idx=E_idx)

        if self.ligand_conditioning:
            V, Y_nodes, Y_edges, Y_m = self._build_ligand_context_features(batch)
        else:
            V = None
            Y_nodes = None
            Y_edges = None
            Y_m = None

        return E, E_idx, V, Y_nodes, Y_edges, Y_m, D_neighbors

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
        E = torch.cat((E_positional, RBF_backbone), -1)
        E = self.protein_edge_embedding(E)
        return self.norm_protein_edges(E)

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

    def _build_ligand_context_features(self, batch):
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
        Y_t_features, Y_t_embedded = self._embed_ligand_atom_types(Y_t)
        V = self._embed_ligand_interaction_features(
            Y=Y,
            Y_t_embedded=Y_t_embedded,
            noised_backbone_pseudo_cb_coords=noised_backbone_pseudo_cb_coords,
        )
        Y_nodes, Y_edges = self._embed_ligand_subgraph_features(
            Y=Y,
            Y_m=Y_m,
            Y_t_features=Y_t_features,
            Y_atom_features=Y_atom_features,
            Y_aromatic=Y_aromatic,
        )
        return V, Y_nodes, Y_edges, Y_m

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
        if self.use_ligand_formal_charge:
            ligand_formal_charge = batch["atom_charge"].float()
            ligand_atom_features["formal_charge"] = (ligand_formal_charge * ligand_mask).unsqueeze(-1)
        if self.use_ligand_aromatic_atom_feature or self.use_ligand_aromatic_edge_feature:
            ligand_aromatic = batch["atom_is_aromatic"].float() * ligand_mask
            if self.use_ligand_aromatic_atom_feature:
                ligand_atom_features["aromatic_atom"] = ligand_aromatic.unsqueeze(-1)
        if self.use_ligand_chirality_tag:
            ligand_chirality = batch["atom_chirality_tag"].long().clamp(min=0, max=2)
            ligand_chirality = F.one_hot(ligand_chirality, num_classes=3).float()
            ligand_atom_features["chirality_tag"] = ligand_chirality * ligand_mask.unsqueeze(-1)
        return ligand_atom_features, ligand_aromatic

    def _gather_ligand_atom_features(self, ligand_atom_features, Y_idx, B, N, device):
        gathered_features = {}
        feature_specs = (
            ("formal_charge", self.ligand_formal_charge_linear, 1),
            ("aromatic_atom", self.ligand_aromatic_atom_linear, 1),
            ("chirality_tag", self.ligand_chirality_tag_linear, self.ligand_chirality_feature_dim),
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
        # Atom type information for context atoms  # Todo: handle Lanthanide metals properly
        Y_t = Y_t.long()
        Y_t_g = torch.tensor(PERIODIC_TABLE_FEATURES[1], device=Y_t.device)[Y_t]
        Y_t_p = torch.tensor(PERIODIC_TABLE_FEATURES[2], device=Y_t.device)[Y_t]
        Y_t_g_1hot_ = torch.nn.functional.one_hot(Y_t_g, 19)
        Y_t_p_1hot_ = torch.nn.functional.one_hot(Y_t_p, 8)
        Y_t_1hot_ = torch.nn.functional.one_hot(Y_t, 120)
        Y_t_features = torch.cat([Y_t_1hot_, Y_t_g_1hot_, Y_t_p_1hot_], -1).float()
        Y_t_embedded = self.type_linear(Y_t_features)
        return Y_t_features, Y_t_embedded

    def _embed_ligand_interaction_features(self, Y, Y_t_embedded, noised_backbone_pseudo_cb_coords):
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
        return self.norm_nodes(V)

    def _embed_ligand_subgraph_features(self, Y, Y_m, Y_t_features, Y_atom_features, Y_aromatic):
        Y_nodes = self.y_nodes(Y_t_features)
        if self.ligand_formal_charge_linear is not None:
            Y_nodes = Y_nodes + self.ligand_formal_charge_linear(Y_atom_features["formal_charge"].float())
        if self.ligand_aromatic_atom_linear is not None:
            Y_nodes = Y_nodes + self.ligand_aromatic_atom_linear(Y_atom_features["aromatic_atom"].float())
        if self.ligand_chirality_tag_linear is not None:
            Y_nodes = Y_nodes + self.ligand_chirality_tag_linear(Y_atom_features["chirality_tag"].float())
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
        Y_edges = self.norm_y_edges(Y_edges)
        return Y_nodes, Y_edges


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

        Y_r = Y.unsqueeze(1).repeat(1, CB.shape[1], 1, 1)
        Y_t_r = Y_t.unsqueeze(1).repeat(1, CB.shape[1], 1)
        Y_m_r = Y_m.unsqueeze(1).repeat(1, CB.shape[1], 1)

        # Y_r = Y[None, :, :].repeat(CB.shape[0], 1, 1)
        # Y_t_r = Y_t[None, :].repeat(CB.shape[0], 1)
        # Y_m_r = Y_m[None, :].repeat(CB.shape[0], 1)

        Y_tmp = torch.gather(Y_r, 2, nn_idx[:, :, :, None].repeat(1, 1, 1, 3))
        Y_t_tmp = torch.gather(Y_t_r, 2, nn_idx)
        Y_m_tmp = torch.gather(Y_m_r, 2, nn_idx)

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
        atom_features_r = atom_features.unsqueeze(1).repeat(1, nn_idx.shape[1], 1, 1)
        gathered_tmp = torch.gather(
            atom_features_r,
            2,
            nn_idx[:, :, :, None].repeat(1, 1, 1, feature_dim),
        )
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
