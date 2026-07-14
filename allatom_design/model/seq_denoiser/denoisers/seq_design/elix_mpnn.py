import math
from functools import partial
from typing import Optional, Union
from omegaconf import DictConfig

import numpy as np
import torch
import torch.nn as nn
import torch._dynamo as dynamo
from torch.nn import functional as F
from torchtyping import TensorType

import allatom_design.data.const as const
import allatom_design.model.seq_denoiser.denoisers.seq_design.potts as potts
from allatom_design.model.seq_denoiser.denoisers.sidechain_prediction import (
    ChiAnglePredictionHead,
)
from allatom_design.model.seq_denoiser.denoisers.seq_design.mpnn_utils import (
    cat_neighbors_nodes,
    gather_nodes,
)
from allatom_design.model.seq_denoiser.denoisers.seq_design.tokenfeatures import (
    PositionalEncodings,
    TokenFeatures,
)
from allatom_design.utils.tensor_utils import batched_gather
from atomworks.constants import (
    DNA_BACKBONE_ATOM_NAMES,
    ELEMENT_NAME_TO_ATOMIC_NUMBER,
    NUCLEIC_ACID_BACKBONE_ATOM_NAMES,
    PROTEIN_BACKBONE_ATOM_NAMES,
    RNA_BACKBONE_ATOM_NAMES,
    STANDARD_AA,
    STANDARD_AA_TIP_ATOM_NAMES,
    STANDARD_DNA,
    STANDARD_PURINE_RESIDUES,
    STANDARD_PYRIMIDINE_RESIDUES,
    STANDARD_RNA,
    UNKNOWN_AA,
    UNKNOWN_DNA,
    UNKNOWN_RNA,
)


class ElixMPNN(nn.Module):
    """Modified ProteinMPNN network to predict sequence from full atom structure."""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.ligand_conditioning = cfg.ligand_conditioning
        self.hidden_dim = cfg.hidden_dim
        self.num_encoder_layers = cfg.num_encoder_layers
        self.num_decoder_layers = cfg.num_decoder_layers
        self.use_mpnn_decoder = cfg.get("use_mpnn_decoder", True)
        self.k_neighbors = cfg.k_neighbors
        self.use_potts_encoding = bool(cfg.get("use_potts_encoding", False))
        self.sequence_encoding = const.POTTS_ENCODING if self.use_potts_encoding else const.AF3_ENCODING
        self.n_tokens = self.sequence_encoding.n_tokens
        self.expansion_mode = cfg.get("expansion_mode", None)
        self.use_context_skip_connection = cfg.get("use_context_skip_connection", False)
        self.use_potts_context_skip_concat = self.expansion_mode == "node_concat_context_skip"
        if self.use_context_skip_connection:
            assert self.ligand_conditioning, (
                "use_context_skip_connection requires ligand_conditioning=True; "
                "the skip path sources its signal from the ligand ContextModule."
            )
        if self.use_potts_context_skip_concat:
            assert self.ligand_conditioning, (
                "expansion_mode='node_concat_context_skip' requires ligand_conditioning=True; "
                "the Potts edge expansion sources its skip signal from the ligand ContextModule."
            )
        self.return_context_skip = self.use_context_skip_connection or self.use_potts_context_skip_concat

        self.token_features = TokenFeatures(cfg.token_features)
        self.W_e = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) # Edge embedding
        self.W_s = nn.Linear(self.n_tokens, self.hidden_dim, bias=False) # Sequence embedding
        self.dropout = nn.Dropout(cfg.dropout_p)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(self.hidden_dim, self.hidden_dim*3, dropout=cfg.dropout_p,
                     is_last_layer=(i == self.num_encoder_layers - 1))
            for i in range(self.num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(self.hidden_dim, self.hidden_dim*3, dropout=cfg.dropout_p,
                     use_context_skip_connection=self.use_context_skip_connection)
            for i in range(self.num_decoder_layers)
        ])

        if self.ligand_conditioning:
            cfg_lmpnn_module = cfg.get("lmpnn_module", None)
            self.num_context_feature_processor_layers = cfg_lmpnn_module.get("num_context_feature_processor_layers", None)
            self.num_context_feature_aggregator_layers = cfg_lmpnn_module.get("num_context_feature_aggregator_layers", None)

            assert cfg_lmpnn_module is not None, "lmpnn_module is required for ligand conditioning"
            assert self.num_context_feature_processor_layers is not None, "num_context_feature_processor_layers is required for ligand conditioning"
            assert self.num_context_feature_aggregator_layers is not None, "num_context_feature_aggregator_layers is required for ligand conditioning"

            self.context_edge_update = cfg_lmpnn_module.get("context_edge_update", False)
            context_module_dropout_p = float(
                cfg_lmpnn_module.get("dropout_p", cfg.dropout_p)
            )

            # Encapsulate context feature processing into a separate module
            self.context_module = ContextModule(
                hidden_dim=self.hidden_dim,
                dropout_p=context_module_dropout_p,
                num_processor_layers=self.num_context_feature_processor_layers,
                num_aggregator_layers=self.num_context_feature_aggregator_layers,
                context_edge_update=self.context_edge_update,
                return_context_skip=self.return_context_skip,
            )

        # Potts decoder
        self.use_potts = self.cfg.potts.use_potts
        self.sidechain_prediction_cfg = cfg.get("sidechain_prediction", {})
        self.use_sidechain_prediction = bool(self.sidechain_prediction_cfg.get("enabled", False))
        if self.use_sidechain_prediction and not self.use_potts:
            raise ValueError("sidechain_prediction.enabled=true requires potts.use_potts=true")

        if self.use_potts:
            self.k_neighbors_potts = cfg.potts.get("k_neighbors_potts", None)
            self.max_dist_potts = cfg.potts.get("max_dist_potts", None)
            self.parameterization = cfg.potts.parameterization
            self.num_factors = cfg.potts.num_factors
            self.num_heads = cfg.potts.get("num_heads", None)
            self.reduce = cfg.potts.get("reduce", "mean")
            self.norm_potts_inputs = cfg.potts.get("norm_potts_inputs", False)

            self.dim_multi_head = None
            if self.expansion_mode == "node_concat":
                self.dim_nodes_potts = self.hidden_dim
                self.dim_edges_potts = self.hidden_dim * 3
            elif self.expansion_mode == "node_concat_context_skip":
                if self.parameterization != "factor":
                    raise ValueError(
                        "expansion_mode='node_concat_context_skip' requires "
                        "potts.parameterization='factor'; "
                        f"got {self.parameterization!r}"
                    )
                self.dim_nodes_potts = self.hidden_dim * 2
                self.dim_edges_potts = self.hidden_dim * 5
            elif self.expansion_mode is None:
                self.dim_nodes_potts = self.dim_edges_potts = self.hidden_dim
            elif self.expansion_mode == "multi_head_factor":
                if self.parameterization != "multi_head_factor":
                    raise ValueError(
                        "expansion_mode='multi_head_factor' requires "
                        "potts.parameterization='multi_head_factor'; "
                        f"got {self.parameterization!r}"
                    )
                self.dim_nodes_potts = self.dim_edges_potts = self.hidden_dim
                self.dim_multi_head = cfg.potts.multi_head.get("dim_multi_head", 128)
                self.num_heads = cfg.potts.multi_head.get("num_heads", 4)
                self.reduce = cfg.potts.multi_head.get("reduce", "mean")
            elif self.expansion_mode == "node_concat_multi_head":
                if self.parameterization != "multi_head_factor":
                    raise ValueError(
                        "expansion_mode='node_concat_multi_head' requires "
                        "potts.parameterization='multi_head_factor'; "
                        f"got {self.parameterization!r}"
                    )
                self.dim_nodes_potts = self.hidden_dim
                self.dim_edges_potts = self.hidden_dim * 3
                self.dim_multi_head = cfg.potts.multi_head.get("dim_multi_head", 128)
                self.num_heads = cfg.potts.multi_head.get("num_heads", 4)
                self.reduce = cfg.potts.multi_head.get("reduce", "mean")
            else:
                raise ValueError(
                    f"Invalid expansion mode: {self.expansion_mode}. Must be "
                    "'node_concat', 'node_concat_context_skip', "
                    "'multi_head_factor', or 'node_concat_multi_head'."
                )

            if self.norm_potts_inputs:
                self.norm_potts_inputs_nodes = nn.LayerNorm(self.dim_nodes_potts)
                self.norm_potts_inputs_edges = nn.LayerNorm(self.dim_edges_potts)

            potts_init = partial(potts.GraphPotts,
                dim_nodes=self.dim_nodes_potts,
                dim_edges=self.dim_edges_potts,
                dim_multi_head=self.dim_multi_head,
                num_states=self.n_tokens,
                parameterization=self.parameterization,
                num_factors=self.num_factors,
                num_heads=self.num_heads,
                reduce=self.reduce,
                symmetric_J=cfg.potts.symmetric_J,
                dropout=cfg.dropout_p,
            )
            self.decoder_S_potts = potts_init()

            if self.use_sidechain_prediction:
                sidechain_hidden_dim = int(self.sidechain_prediction_cfg.get("hidden_dim", self.hidden_dim))
                self.chi_angle_prediction_head = ChiAnglePredictionHead(
                    input_dim=self.dim_nodes_potts,
                    hidden_dim=sidechain_hidden_dim,
                    num_chi=int(self.sidechain_prediction_cfg.get("num_chi", 4)),
                    num_bins=int(self.sidechain_prediction_cfg.get("num_bins", 72)),
                    dropout_p=float(self.sidechain_prediction_cfg.get("dropout_p", cfg.dropout_p)),
                    use_node_norm=bool(self.sidechain_prediction_cfg.get("use_node_norm", True)),
                )
            else:
                self.chi_angle_prediction_head = None

        # Output layers
        self.W_out = nn.Linear(self.hidden_dim, self.n_tokens, bias=True)

        # Initialize weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Skip path: zero-init W_ctx so the skip contribution starts at 0 (ControlNet-style).
        # The model initializes close to the use_context_skip_connection=False baseline,
        # and the skip path only learns non-trivial contributions if they reduce loss.
        if self.use_context_skip_connection:
            for layer in self.decoder_layers:
                nn.init.zeros_(layer.W_ctx.weight)
                nn.init.zeros_(layer.W_ctx.bias)


    def forward(self, batch: dict[str, TensorType["b ..."]], is_sampling: bool):
        # Get token-level features
        B, N, C = batch["restype"].shape
        if C != self.n_tokens:
            raise ValueError(
                f"ElixMPNN expected restype alphabet size {self.n_tokens}, got {C}. "
                "Check denoiser.mpnn.use_potts_encoding and restype projection."
            )
        h_V = torch.zeros((B, N, self.hidden_dim), device=batch["restype"].device)

        # Concatenate residue-level features to h_V
        if self.use_potts_encoding:
            restype = batch["restype"]
        else:
            ## first, mask out residues using gap token
            masked = F.one_hot(torch.full((B, N), const.AF3_ENCODING.token_to_idx["<G>"],
                                          device=batch["restype"].device), num_classes=C).float()

            #! (JH) During sampling, seq_cond_mask is also 1 for padded tokens
            #! (JH) So padded parts are also considered as gaps here, but I guess it's okay.
            restype = torch.where(batch["seq_cond_mask"].unsqueeze(-1).bool(), batch["restype"], masked)
        h_S = self.W_s(restype) #! (JH) different from the original lmpnn (zero-initialized)

        # Build graph and get edge features
        h_E, E_idx, V, Y_nodes, Y_edges, Y_m, D_neighbors = self.token_features(batch)
        #! (JH) h_E and E_idx are also considering ligand atoms here.
        #! (JH) but h_E and E_idx are masked out for padded tokens (token_exists_mask is 0 for padded tokens)

        # Prepare protein residue node mask
        protein_residue_node_mask = batch["protein_residue_node_mask"]
        protein_residue_node_mask_2d = gather_nodes(protein_residue_node_mask.unsqueeze(-1), E_idx).squeeze(-1)
        protein_residue_node_mask_2d = protein_residue_node_mask.unsqueeze(-1) * protein_residue_node_mask_2d

        # Pass through encoder layers
        # Residue-level encoding, for standard AAs in protein chains only
        h_V = h_V + h_S
        h_E = self.W_e(h_E)

        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, protein_residue_node_mask, protein_residue_node_mask_2d)

        # Process ligand context features
        h_V_C_skip = None
        if self.ligand_conditioning:
            h_V, h_V_C_skip = self.context_module(
                h_V=h_V,
                h_E=h_E,
                V=V,
                Y_nodes=Y_nodes,
                Y_edges=Y_edges,
                Y_m=Y_m,
                E_idx=E_idx,
                protein_residue_node_mask=protein_residue_node_mask,
            )

        # Add residue-level features to the encoded features before passing through decoder layers
        if self.use_mpnn_decoder:
            h_V = h_V + h_S
            for layer in self.decoder_layers:
                h_V, h_E = layer(h_V = h_V, h_E = h_E,
                                    mask_V = protein_residue_node_mask, E_idx = E_idx,
                                    mask_attend = protein_residue_node_mask_2d, h_V_C_skip=h_V_C_skip)

        h_V_potts = self._expand_potts_nodes(h_V, h_V_C_skip)
        h_E = self._expand_potts_edges(h_V, h_E, E_idx, h_V_C_skip)

        # Potts model
        sidechain_prediction_aux = None
        if self.use_potts:
            if self.norm_potts_inputs:
                h_V_potts = self.norm_potts_inputs_nodes(h_V_potts)
                h_E = self.norm_potts_inputs_edges(h_E)

            if self.chi_angle_prediction_head is not None:
                sidechain_prediction_aux = self.chi_angle_prediction_head(
                    h_V_potts,
                    chi_angles=batch.get("chi_angles", None) if not is_sampling else None,
                )

            if self.max_dist_potts is not None:
                protein_residue_node_mask_2d = protein_residue_node_mask_2d * (D_neighbors <= self.max_dist_potts)  # mask out edges that are too far away

            if self.k_neighbors_potts is not None:
                # truncate to k_neighbors_potts
                h_E = h_E[:, :, :self.k_neighbors_potts]
                E_idx = E_idx[:, :, :self.k_neighbors_potts]
                protein_residue_node_mask_2d = protein_residue_node_mask_2d[:, :, :self.k_neighbors_potts]

            h, J = self.decoder_S_potts(h_V_potts, h_E, E_idx, protein_residue_node_mask, protein_residue_node_mask_2d)
            coupling_mask = potts.build_coupling_mask(
                E_idx,
                protein_residue_node_mask,
                protein_residue_node_mask_2d,
                require_reciprocal=self.decoder_S_potts.symmetric_J,
            )
            potts_decoder_aux = {
                "h": h,
                "J": J,
                "edge_idx": E_idx,
                "mask_i": protein_residue_node_mask,
                "mask_ij": protein_residue_node_mask_2d,
                "coupling_mask": coupling_mask,
            }

        logits = self.W_out(h_V)

        # Output features
        mpnn_feature_dict = {"h_V": h_V, "h_ESV": h_E, "E_idx": E_idx} # Todo: Need to change "h_ESV" to "h_E" in the pipeline later
        if self.use_potts:
            mpnn_feature_dict["potts_decoder_aux"] = potts_decoder_aux
        if sidechain_prediction_aux is not None:
            mpnn_feature_dict["sidechain_prediction_aux"] = sidechain_prediction_aux

        return logits, mpnn_feature_dict

    def _expand_potts_nodes(self, h_V, h_V_C_skip=None):
        if self.expansion_mode != "node_concat_context_skip":
            return h_V
        if h_V_C_skip is None:
            raise RuntimeError(
                "expansion_mode='node_concat_context_skip' requires h_V_C_skip "
                "from ContextModule; check ligand_conditioning and return_context_skip."
            )
        return torch.cat([h_V, h_V_C_skip], -1)

    def _expand_potts_edges(self, h_V, h_E, E_idx, h_V_C_skip=None):
        if self.expansion_mode not in (
            "node_concat",
            "node_concat_context_skip",
            "node_concat_multi_head",
        ):
            return h_E

        h_E_neighbors = cat_neighbors_nodes(h_V, h_E, E_idx) # [h_E_ij, h_V_j]
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E_neighbors.size(-2), -1)
        h_E_expanded = torch.cat([h_V_expand, h_E_neighbors], -1)

        if self.expansion_mode != "node_concat_context_skip":
            return h_E_expanded

        if h_V_C_skip is None:
            raise RuntimeError(
                "expansion_mode='node_concat_context_skip' requires h_V_C_skip "
                "from ContextModule; check ligand_conditioning and return_context_skip."
            )
        h_V_C_skip_j = gather_nodes(h_V_C_skip, E_idx)
        h_V_C_skip_i = h_V_C_skip.unsqueeze(-2).expand(
            -1, -1, h_E_neighbors.size(-2), -1
        )
        return torch.cat([h_E_expanded, h_V_C_skip_i, h_V_C_skip_j], -1)


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30, is_last_layer=False):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.is_last_layer = is_last_layer

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

        if not self.is_last_layer:
            # only initialize if not last layer to avoid unused parameters
            self.W11 = nn.Linear(num_in, num_hidden, bias=True)
            self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
            self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
            self.norm3 = nn.LayerNorm(num_hidden)


    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx) # V_j -> E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1) # V_i -> E_ij. [B, L, K, 3*num_hidden]

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV))))) # [B, L, K, num_hidden]

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message

        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        # Edge updates
        if not self.is_last_layer:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx) # V_j -> E_ij
            h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
            h_EV = torch.cat([h_V_expand, h_EV], -1) # V_i -> E_ij. [B, L, K, 3*num_hidden]
            h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV))))) # [B, L, K, num_hidden]
            h_E = self.norm3(h_E + self.dropout3(h_message))
            if mask_attend is not None:
                h_E = mask_attend.unsqueeze(-1) * h_E

        return h_V, h_E

class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30, is_last_layer=False, use_context_skip_connection=False):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.is_last_layer = is_last_layer
        self.use_context_skip_connection = use_context_skip_connection

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

        # only initialize if not last layer to avoid unused parameters
        self.W11 = nn.Linear(num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.norm3 = nn.LayerNorm(num_hidden)

        if self.use_context_skip_connection:
            self.W_ctx = nn.Linear(num_hidden, num_hidden, bias=True)
            self.norm_ctx = nn.LayerNorm(num_hidden)
            self.dropout_ctx = nn.Dropout(dropout)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None, h_V_C_skip=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx) # V_j -> E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1) # V_i -> E_ij. [B, L, K, 3*num_hidden]

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV))))) # [B, L, K, num_hidden]

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message

        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        if self.use_context_skip_connection:
            # Inject ligand-context refinement. Assertion in ElixMPNN.__init__ guarantees h_V_C_skip is provided.
            skip = self.norm_ctx(h_V + self.dropout_ctx(self.W_ctx(h_V_C_skip)))
            if mask_V is not None:
                m = mask_V.unsqueeze(-1)
                h_V = m * skip + (1.0 - m) * h_V
            else:
                h_V = skip

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        # Edge updates
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx) # V_j -> E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1) # V_i -> E_ij. [B, L, K, 3*num_hidden]
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV))))) # [B, L, K, num_hidden]
        h_E = self.norm3(h_E + self.dropout3(h_message))
        if mask_attend is not None:
            h_E = mask_attend.unsqueeze(-1) * h_E

        return h_V, h_E

class ContextModule(nn.Module):
    def __init__(self, hidden_dim: int, dropout_p: float, num_processor_layers: int, num_aggregator_layers: int, context_edge_update: bool,
                 return_context_skip: bool):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_edge_update = context_edge_update
        self.return_context_skip = return_context_skip

        # Projections
        self.W_v = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.W_c = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.W_nodes_y = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.W_edges_y = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.V_C = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.V_C_norm = torch.nn.LayerNorm(self.hidden_dim)
        self.dropout = torch.nn.Dropout(dropout_p)

        # Stacks
        self.context_feature_processor = torch.nn.ModuleList(
            [Contextfeatureprocessor(self.hidden_dim, 3 * self.hidden_dim,
                                     dropout=dropout_p,
                                     context_edge_update=self.context_edge_update) for i in range(num_processor_layers)]

        )

        self.context_feature_aggregator = torch.nn.ModuleList(
            [Contextfeatureaggregator(self.hidden_dim, 3 * self.hidden_dim,
                                      dropout=dropout_p,
                                      context_edge_update=self.context_edge_update,
                                      is_last_layer=(i == num_aggregator_layers - 1)) for i in range(num_aggregator_layers)]
        )

    # @dynamo.disable()
    def forward(self, h_V = None, h_E = None,
                V = None, Y_nodes = None, Y_edges = None,
                Y_m = None, E_idx = None,
                protein_residue_node_mask = None):
        h_E_context = self.W_v(V)
        h_V_C = self.W_c(h_V)
        Y_m_edges = Y_m[:, :, :, None] * Y_m[:, :, None, :]
        Y_nodes = self.W_nodes_y(Y_nodes)
        Y_edges = self.W_edges_y(Y_edges)

        if not self.context_edge_update:
            for i in range(len(self.context_feature_aggregator)):
                Y_nodes, _ = self.context_feature_processor[i](
                    h_V=Y_nodes, h_E=Y_edges, mask_V=Y_m, mask_attend=Y_m_edges,
                )

                h_V_C, _ = self.context_feature_aggregator[i](
                    h_V=h_V_C, h_E_context=h_E_context, Y_nodes = Y_nodes,
                    mask_V=protein_residue_node_mask, mask_attend=Y_m
                )
        else:
            for i in range(len(self.context_feature_aggregator)):
                Y_nodes, Y_edges = self.context_feature_processor[i](
                    h_V=Y_nodes, h_E=Y_edges, mask_V=Y_m, mask_attend=Y_m_edges) # Overwrite Y_edges with the new context features

                h_V_C, h_E_context = self.context_feature_aggregator[i](
                    h_V=h_V_C, h_E_context=h_E_context, Y_nodes = Y_nodes, mask_V=protein_residue_node_mask, mask_attend=Y_m
                ) # Overwrite h_E_context with the new context features

        h_V_C_skip = h_V_C if self.return_context_skip else None

        h_V_C = self.V_C(h_V_C)
        h_V = h_V + self.V_C_norm(self.dropout(h_V_C))
        return h_V, h_V_C_skip

class Contextfeatureprocessor(nn.Module): # self.y_context_encoder_layers in ligandMPNN
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None,
                 scale=30, context_edge_update=False):
        super(Contextfeatureprocessor, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.context_edge_update = context_edge_update
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.num_hidden)
        self.norm2 = nn.LayerNorm(self.num_hidden)

        # Node update
        self.W1 = nn.Linear(self.num_in, self.num_hidden, bias=True) # Following the foundry's LigandMPNN implementation
        self.W2 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)
        self.W3 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)

        # Edge update
        if self.context_edge_update:
            self.W11 = nn.Linear(self.num_in, self.num_hidden, bias=True) # nh * 2 for vi AND vj
            self.W12 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)
            self.W13 = nn.Linear(self.num_hidden, self.num_hidden, bias=True) # num_in is hidden dim of edges h_E
            self.norm3 = nn.LayerNorm(self.num_hidden)
            self.dropout3 = nn.Dropout(dropout)

        # Activation and feedforward
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(self.num_hidden, num_hidden * 4)

    # @dynamo.disable()
    def forward(self, h_V = None, h_E = None, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Source node features
        h_V_i = h_V.unsqueeze(-2).expand(-1,-1,-1,h_E.size(-2),-1) # [B, L, M, M, C_node]

        # Destination node features
        h_V_j = h_V.unsqueeze(-3).expand(-1, -1, h_E.size(-3), -1, -1)  # [B, L, M, M, C_node]

        h_EV = torch.cat([h_V_i, h_E, h_V_j], -1) # [B, L, M, M, C_edge + C_node + C_node]
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message

        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        if self.context_edge_update:
            h_V_i = h_V.unsqueeze(-2).expand(-1, -1, -1, h_E.size(-2), -1)
            h_V_j = h_V.unsqueeze(-3).expand(-1, -1, h_E.size(-3), -1, -1)
            h_EV = torch.cat([h_V_i, h_E, h_V_j], -1)
            h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
            h_E = self.norm3(h_E + self.dropout3(h_message))

            if mask_attend is not None:
                h_E = mask_attend.unsqueeze(-1) * h_E

            return h_V, h_E
        else:
            return h_V, None

class Contextfeatureaggregator(nn.Module): #! (JH) self.context_encoder_layers in ligandMPNN
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None,
                 scale=30, context_edge_update=False, is_last_layer=False):
        super(Contextfeatureaggregator, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.context_edge_update = context_edge_update
        self.is_last_layer = is_last_layer

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.num_hidden)
        self.norm2 = nn.LayerNorm(self.num_hidden)

        # Node update
        self.W1 = nn.Linear(self.num_in, self.num_hidden, bias=True)
        self.W2 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)
        self.W3 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)

        # Edge update
        if self.context_edge_update and not self.is_last_layer:
            self.W11 = nn.Linear(self.num_in, self.num_hidden, bias=True)
            self.W12 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)
            self.W13 = nn.Linear(self.num_hidden, self.num_hidden, bias=True) # num_in is hidden dim of edges h_E
            self.norm3 = nn.LayerNorm(self.num_hidden)
            self.dropout3 = nn.Dropout(dropout)

        # Activation and feedforward
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(self.num_hidden, num_hidden * 4)

    # @dynamo.disable()
    def forward(self, h_V = None, h_E_context = None, Y_nodes = None, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate Y_nodes to h_E_context (add ligand node features edges)
        h_E_context_cat = torch.cat([h_E_context, Y_nodes], -1) #Y_i -> E_ij. [B, L, K, 2*num_hidden]
        # concatenate h_V to h_E_context_cat (add protein node features to edges)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E_context_cat.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E_context_cat], -1) #V_i -> E_ij. [B, L, K, 3*num_hidden]

        # Run message passing
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        #! h_message here is the context features for each protein node

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message

        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        # edge updates
        if self.context_edge_update and not self.is_last_layer:
            h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E_context_cat.size(-2),-1)
            h_EV = torch.cat([h_V_expand, h_E_context_cat], -1) #V_i -> E_ij. [B, L, K, 3*num_hidden]
            # h_E_context_cat is not updated but h_V is updated, so we only need to concatenate h_V to h_E_context_cat

            # Run message passing
            h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
            h_E_context = self.norm3(h_E_context + self.dropout3(h_message))

            if mask_attend is not None:
                h_E_context = mask_attend.unsqueeze(-1) * h_E_context

            return h_V, h_E_context
        else:
            return h_V, None

class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = torch.nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = torch.nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

def get_tokenwise_coords(coords: TensorType["b n_atoms 3", float],
                         tokenwise_atom_idxs: TensorType["b n_tokens"],
                         tokenwise_atom_idxs_mask: TensorType["b n_tokens"],
                         ) -> TensorType["b n_tokens MAX_NUM_ATOMS 3", float]:
    """
    Get token-level coordinates (padded to max_num_atoms per token). Batched version of pad_atom_feats_to_tokenwise for just coords.
    tokenwise_atom_idxs_mask is basically token_pad_mask for MAX_NUM_ATOMS atoms per token.
    """

    tokenwise_coords = batched_gather(coords, tokenwise_atom_idxs, dim=1, no_batch_dims=1) * tokenwise_atom_idxs_mask[..., None]

    return tokenwise_coords

def get_atomwise_coords(
    batch: dict[str, TensorType["b ..."]],
    tokenwise_coords: TensorType["b n_tokens 23 3", float],
) -> TensorType["b n_atoms 3", float]:
    """
    Inverse of get_tokenwise_coords. Given tokenwise coords [B, n_tokens, max_num_atoms, 3],
    reconstruct atomwise coords [B, n_atoms, 3].
    """
    B = batch["coords"].shape[0]
    device = batch["coords"].device

    x = batch["atomwise_token_idx"] * tokenwise_coords.shape[-2]  # flattened atomwise token indices
    is_start = torch.ones_like(x, dtype=torch.bool)
    is_start[:, 1:] = x[:, 1:] != x[:, :-1]
    pos = torch.arange(x.shape[-1], device=x.device).unsqueeze(0).expand(B, x.shape[-1])
    start_pos = torch.where(is_start, pos, torch.full_like(pos, -1))
    first_pos = torch.cummax(start_pos, dim=1).values
    local_idx = pos - first_pos
    gather_idx = x + local_idx
    gather_idx = gather_idx

    new_coords = batched_gather(tokenwise_coords.view(B, -1, 3), gather_idx, dim=1, no_batch_dims=1)
    return new_coords


##############
class Adapter(nn.Module):
    def __init__(self, in_dim, out_dim=None, expansion=4, dropout=0.1):
        super().__init__()
        out_dim = out_dim or in_dim



        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        self.norm0 = nn.LayerNorm(out_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, expansion * out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * out_dim, out_dim),
        )
        # 안정성: 처음에는 거의 identity/projection만 보이게
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        y = self.norm0(self.proj(x))
        return y + self.mlp(y)
