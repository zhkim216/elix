from omegaconf import DictConfig, ListConfig

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchtyping import TensorType

import allatom_design.data.const as const
import allatom_design.model.seq_denoiser.denoisers.seq_design.potts as potts
from allatom_design.model.seq_denoiser.denoisers.sidechain_prediction import (
    ChiAnglePredictionHead,
)
from allatom_design.model.seq_denoiser.denoisers.seq_design.mpnn_layers import (
    CalibyDecLayer,
    ContextModule,
    Contextfeatureaggregator,
    Contextfeatureprocessor,
    DecLayer,
    EncLayer,
    PositionWiseFeedForward,
)
from allatom_design.model.seq_denoiser.denoisers.seq_design.mpnn_utils import (
    cat_neighbors_nodes,
    gather_nodes,
)
from allatom_design.model.seq_denoiser.denoisers.seq_design.sparse_triangle_multiplication import (
    SparseProteinPairTriangleMultiplication,
    SparseSharedAtomTriangleMultiplication,
    build_protein_pair_edge_slot_matches,
    build_shared_atom_lookup,
)
from allatom_design.model.seq_denoiser.denoisers.seq_design.tokenfeatures import (
    TokenFeatures,
)


def _validated_layer_indices(
    config: DictConfig | dict,
    key: str,
    *,
    num_layers: int,
    config_path: str,
) -> tuple[int, ...]:
    """Read a unique, in-range list of layer indices from a config block."""
    raw_indices = config.get(key, []) or []
    if not isinstance(raw_indices, (list, tuple, ListConfig)):
        raise ValueError(f"{config_path}.{key} must be a list of layer indices")

    layer_indices: list[int] = []
    for raw_index in raw_indices:
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError(
                f"{config_path}.{key} must contain only integer layer indices; "
                f"got {raw_index!r}"
            )
        layer_index = int(raw_index)
        if not 0 <= layer_index < num_layers:
            raise ValueError(
                f"{config_path}.{key} contains layer {layer_index}, but valid "
                f"indices are 0..{num_layers - 1}"
            )
        layer_indices.append(layer_index)

    if len(set(layer_indices)) != len(layer_indices):
        raise ValueError(f"{config_path}.{key} must not contain duplicates")
    if layer_indices != sorted(layer_indices):
        raise ValueError(f"{config_path}.{key} must be sorted")
    return tuple(layer_indices)


class ElixMPNN(nn.Module):
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
        self.decoder_input_mode = str(cfg.get("decoder_input_mode", "legacy_node_add"))
        if self.decoder_input_mode not in {
            "legacy_node_add",
            "caliby_concat",
        }:
            raise ValueError(
                "Invalid decoder_input_mode: "
                f"{self.decoder_input_mode!r}. Expected 'legacy_node_add' or "
                "'caliby_concat'."
            )
        self.use_caliby_decoder = self.decoder_input_mode == "caliby_concat"
        if self.use_caliby_decoder and not self.use_mpnn_decoder:
            raise ValueError(
                f"decoder_input_mode={self.decoder_input_mode!r} requires use_mpnn_decoder=true"
            )
        requested_final_encoder_edge_update = bool(
            cfg.get("update_final_encoder_edge", False)
        )
        self.update_final_encoder_edge = (
            self.use_caliby_decoder or requested_final_encoder_edge_update
        )
        self.expansion_mode = cfg.get("expansion_mode", None)
        self.use_shared_edge_multi_head_potts = (
            self.expansion_mode
            == "edge_node_context_concat_multi_head_gate_nonlinear"
        )
        if (
            self.expansion_mode not in {None, "node_concat"}
            and not self.use_shared_edge_multi_head_potts
        ):
            raise ValueError(
                f"Invalid expansion mode: {self.expansion_mode!r}. Expected "
                "'node_concat', the shared-edge gated multi-head mode, or null for "
                "decoder_input_mode='caliby_concat'."
            )
        self.full_multi_head_aggregation = (
            "gate_nonlinear"
            if self.use_shared_edge_multi_head_potts
            else None
        )
        self.use_context_skip_connection = cfg.get("use_context_skip_connection", False)
        shared_atom_triangle_cfg = cfg.get("shared_atom_triangle", {}) or {}
        self.shared_atom_mixing_encoder_layers = _validated_layer_indices(
            shared_atom_triangle_cfg,
            "mixing_encoder_layers",
            num_layers=self.num_decoder_layers,
            config_path="shared_atom_triangle",
        )
        self.use_shared_atom_triangle = bool(
            self.shared_atom_mixing_encoder_layers
        )
        protein_pair_triangle_cfg = cfg.get("protein_pair_triangle", {}) or {}
        self.protein_pair_triangle_mixing_encoder_layers = (
            _validated_layer_indices(
                protein_pair_triangle_cfg,
                "mixing_encoder_layers",
                num_layers=self.num_decoder_layers,
                config_path="protein_pair_triangle",
            )
        )
        self.use_mixing_encoder_pair_triangle = bool(
            self.protein_pair_triangle_mixing_encoder_layers
        )
        if self.use_shared_atom_triangle and (
            not self.ligand_conditioning
            or not self.use_mpnn_decoder
            or self.decoder_input_mode != "legacy_node_add"
        ):
            raise ValueError(
                "shared_atom_triangle.mixing_encoder_layers requires "
                "ligand_conditioning=true, "
                "use_mpnn_decoder=true, and decoder_input_mode='legacy_node_add'"
            )
        if self.use_mixing_encoder_pair_triangle and (
            not self.use_mpnn_decoder
            or self.decoder_input_mode != "legacy_node_add"
        ):
            raise ValueError(
                "protein_pair_triangle.mixing_encoder_layers requires "
                "use_mpnn_decoder=true and decoder_input_mode='legacy_node_add'"
            )
        if self.use_caliby_decoder and self.expansion_mode is not None:
            raise ValueError(
                "decoder_input_mode='caliby_concat' supplies the 3H Potts edge "
                "state directly and requires expansion_mode=null"
            )
        if not self.use_caliby_decoder and self.expansion_mode is None:
            raise ValueError(
                "decoder_input_mode='legacy_node_add' requires "
                "expansion_mode='node_concat' or the shared-edge gated "
                "multi-head mode"
            )
        if self.use_caliby_decoder and self.use_context_skip_connection:
            raise ValueError(
                "decoder_input_mode='caliby_concat' does not support "
                "use_context_skip_connection=true"
            )
        if self.use_shared_edge_multi_head_potts and (
            not self.ligand_conditioning
            or not self.use_mpnn_decoder
            or self.decoder_input_mode != "legacy_node_add"
        ):
            raise ValueError(
                f"expansion_mode={self.expansion_mode!r} requires "
                "ligand_conditioning=true, use_mpnn_decoder=true, and "
                "decoder_input_mode='legacy_node_add'"
            )
        if self.use_context_skip_connection:
            assert self.ligand_conditioning, (
                "use_context_skip_connection requires ligand_conditioning=True; "
                "the skip path sources its signal from the ligand ContextModule."
            )
        self.return_context_skip = (
            self.use_context_skip_connection
            or self.use_shared_edge_multi_head_potts
        )

        self.token_features = TokenFeatures(cfg.token_features)
        self.W_e = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) # Edge embedding
        self.W_s = nn.Linear(self.n_tokens, self.hidden_dim, bias=False) # Sequence embedding
        self.dropout = nn.Dropout(cfg.dropout_p)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(self.hidden_dim, self.hidden_dim*3, dropout=cfg.dropout_p,
                     is_last_layer=(not self.update_final_encoder_edge and i == self.num_encoder_layers - 1))
            for i in range(self.num_encoder_layers)
        ])

        # Decoder layers
        if self.use_caliby_decoder:
            self.decoder_layers = nn.ModuleList([
                CalibyDecLayer(self.hidden_dim, self.hidden_dim * 3, dropout=cfg.dropout_p)
                for _ in range(self.num_decoder_layers)
            ])
        else:
            self.decoder_layers = nn.ModuleList([
                DecLayer(self.hidden_dim, self.hidden_dim*3, dropout=cfg.dropout_p,
                         use_context_skip_connection=self.use_context_skip_connection)
                for _ in range(self.num_decoder_layers)
            ])

        if self.ligand_conditioning:
            cfg_lmpnn_module = cfg.get("lmpnn_module", None)
            self.num_context_feature_processor_layers = cfg_lmpnn_module.get("num_context_feature_processor_layers", None)
            self.num_context_feature_aggregator_layers = cfg_lmpnn_module.get("num_context_feature_aggregator_layers", None)

            assert cfg_lmpnn_module is not None, "lmpnn_module is required for ligand conditioning"
            assert self.num_context_feature_processor_layers is not None, "num_context_feature_processor_layers is required for ligand conditioning"
            assert self.num_context_feature_aggregator_layers is not None, "num_context_feature_aggregator_layers is required for ligand conditioning"

            legacy_context_edge_update = bool(
                cfg_lmpnn_module.get("context_edge_update", False)
            )
            self.context_pair_update = bool(
                cfg_lmpnn_module.get(
                    "context_pair_update",
                    legacy_context_edge_update,
                )
            )
            self.protein_context_pair_update = bool(
                cfg_lmpnn_module.get(
                    "protein_context_pair_update",
                    legacy_context_edge_update,
                )
            )
            self.update_final_context_processor_edge = bool(
                cfg_lmpnn_module.get("update_final_context_processor_edge", True)
            )
            self.update_final_context_aggregator_edge = bool(
                cfg_lmpnn_module.get(
                    "update_final_context_aggregator_edge",
                    False,
                )
            )
            if (
                self.use_shared_atom_triangle
                and not self.protein_context_pair_update
            ):
                raise ValueError(
                    "shared_atom_triangle.mixing_encoder_layers requires "
                    "lmpnn_module.protein_context_pair_update=true"
                )
            context_module_dropout_p = float(
                cfg_lmpnn_module.get("dropout_p", cfg.dropout_p)
            )

            # Encapsulate context feature processing into a separate module
            self.context_module = ContextModule(
                hidden_dim=self.hidden_dim,
                dropout_p=context_module_dropout_p,
                num_processor_layers=self.num_context_feature_processor_layers,
                num_aggregator_layers=self.num_context_feature_aggregator_layers,
                context_pair_update=self.context_pair_update,
                protein_context_pair_update=self.protein_context_pair_update,
                update_final_processor_edge=self.update_final_context_processor_edge,
                update_final_aggregator_edge=(
                    self.update_final_context_aggregator_edge
                ),
                return_context_skip=self.return_context_skip,
            )

        # Independent shared-atom updates at the configured mixing layers.
        self.shared_atom_triangle_layers = nn.ModuleList()
        self._shared_atom_triangle_slot_by_mixing_layer = {
            layer_index: slot
            for slot, layer_index in enumerate(
                self.shared_atom_mixing_encoder_layers
            )
        }
        if self.use_shared_atom_triangle:
            self.shared_atom_triangle_layers = nn.ModuleList(
                [
                    SparseSharedAtomTriangleMultiplication(
                        dim_protein_context=self.hidden_dim,
                        dim_protein_pair=self.hidden_dim,
                        dim_hidden=int(
                            shared_atom_triangle_cfg.get(
                                "hidden_dim",
                                self.hidden_dim,
                            )
                        ),
                        edge_chunk_size=int(
                            shared_atom_triangle_cfg.get(
                                "edge_chunk_size",
                                4,
                            )
                        ),
                        dropout_p=cfg.dropout_p,
                    )
                    for _ in self.shared_atom_mixing_encoder_layers
                ]
            )
        protein_pair_triangle_hidden_dim = int(
            protein_pair_triangle_cfg.get(
                "hidden_dim",
                self.hidden_dim,
            )
        )
        protein_pair_triangle_target_chunk_size = int(
            protein_pair_triangle_cfg.get("target_chunk_size", 4)
        )
        self._protein_pair_triangle_slot_by_mixing_layer = {
            layer_index: slot
            for slot, layer_index in enumerate(
                self.protein_pair_triangle_mixing_encoder_layers
            )
        }
        self.mixing_encoder_outgoing_triangle_layers = nn.ModuleList(
            [
                SparseProteinPairTriangleMultiplication(
                    dim_pair=self.hidden_dim,
                    dim_hidden=protein_pair_triangle_hidden_dim,
                    direction="outgoing",
                    target_chunk_size=protein_pair_triangle_target_chunk_size,
                )
                for _ in self.protein_pair_triangle_mixing_encoder_layers
            ]
        )
        self.mixing_encoder_incoming_triangle_layers = nn.ModuleList(
            [
                SparseProteinPairTriangleMultiplication(
                    dim_pair=self.hidden_dim,
                    dim_hidden=protein_pair_triangle_hidden_dim,
                    direction="incoming",
                    target_chunk_size=protein_pair_triangle_target_chunk_size,
                )
                for _ in self.protein_pair_triangle_mixing_encoder_layers
            ]
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
            self.norm_potts_inputs = cfg.potts.get("norm_potts_inputs", False)
            self.num_heads = None
            self.reduce = "mean"
            self.adapter_hidden_dim = None
            self.shared_edge_dim = None
            if self.use_caliby_decoder or self.expansion_mode == "node_concat":
                if self.parameterization != "factor":
                    raise ValueError(
                        f"expansion_mode={self.expansion_mode!r} requires "
                        "potts.parameterization='factor'; "
                        f"got {self.parameterization!r}"
                    )
                self.dim_nodes_potts = self.hidden_dim
                self.dim_edges_potts = self.hidden_dim * 3
            else:
                if self.parameterization != "multi_head_factor":
                    raise ValueError(
                        f"expansion_mode={self.expansion_mode!r} requires "
                        "potts.parameterization='multi_head_factor'; "
                        f"got {self.parameterization!r}"
                    )
                self.dim_nodes_potts = self.hidden_dim * 2
                self.dim_edges_potts = self.hidden_dim * 5
                multi_head_cfg = cfg.potts.get("multi_head", {}) or {}
                self.num_heads = int(multi_head_cfg.get("num_heads", 2))
                self.reduce = str(multi_head_cfg.get("reduce", "mean"))
                self.adapter_hidden_dim = int(
                    multi_head_cfg.get("adapter_hidden_dim", 640)
                )
                self.shared_edge_dim = multi_head_cfg.get(
                    "shared_edge_dim", None
                )
                if self.shared_edge_dim is None:
                    raise ValueError(
                        f"expansion_mode={self.expansion_mode!r} requires "
                        "potts.multi_head.shared_edge_dim"
                    )

            if self.norm_potts_inputs:
                self.norm_potts_inputs_nodes = nn.LayerNorm(self.dim_nodes_potts)
                self.norm_potts_inputs_edges = nn.LayerNorm(self.dim_edges_potts)

            self.decoder_S_potts = potts.GraphPotts(
                dim_nodes=self.dim_nodes_potts,
                dim_edges=self.dim_edges_potts,
                num_states=self.n_tokens,
                parameterization=self.parameterization,
                num_heads=self.num_heads,
                reduce=self.reduce,
                full_multi_head_aggregation=self.full_multi_head_aggregation,
                adapter_hidden_dim=self.adapter_hidden_dim,
                shared_edge_dim=self.shared_edge_dim,
                symmetric_J=cfg.potts.symmetric_J,
                dropout=cfg.dropout_p,
            )

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

        if self.use_potts and self.full_multi_head_aggregation is not None:
            self.decoder_S_potts.reset_full_multi_head_special_initialization()

        # Skip path: zero-init W_ctx so the skip contribution starts at 0 (ControlNet-style).
        # The model initializes close to the use_context_skip_connection=False baseline,
        # and the skip path only learns non-trivial contributions if they reduce loss.
        if self.use_context_skip_connection:
            for layer in self.decoder_layers:
                nn.init.zeros_(layer.W_ctx.weight)
                nn.init.zeros_(layer.W_ctx.bias)


    def _apply_shared_atom_triangle_after_decoder_layer(
        self,
        *,
        layer_index: int,
        protein_context_pairs: torch.Tensor,
        protein_pairs: torch.Tensor,
        neighbor_idx: torch.Tensor,
        shared_atom_matches: torch.Tensor,
        shared_atom_edge_mask: torch.Tensor,
        protein_pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Update PP pairs from shared-atom PC x PC triangles."""
        triangle_slot = self._shared_atom_triangle_slot_by_mixing_layer[
            layer_index
        ]
        return self.shared_atom_triangle_layers[triangle_slot](
            protein_context_pairs,
            protein_pairs,
            neighbor_idx,
            shared_atom_matches,
            shared_atom_edge_mask=shared_atom_edge_mask,
            protein_pair_mask=protein_pair_mask,
        )


    def _apply_protein_pair_triangles(
        self,
        *,
        outgoing_layer: nn.Module,
        incoming_layer: nn.Module,
        protein_pairs: torch.Tensor,
        protein_pair_lookup: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        outgoing_update = outgoing_layer(
            protein_pairs,
            *protein_pair_lookup[:3],
        )
        protein_pairs = protein_pairs + self.dropout(outgoing_update)
        incoming_update = incoming_layer(
            protein_pairs,
            *protein_pair_lookup,
        )
        return protein_pairs + self.dropout(incoming_update)


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
        token_feature_outputs = self.token_features(
            batch,
            return_context_metadata=self.use_shared_atom_triangle,
        )
        context_metadata = None
        if self.use_shared_atom_triangle:
            (
                h_E,
                E_idx,
                V,
                Y_nodes,
                Y_edges,
                Y_m,
                D_neighbors,
                context_metadata,
            ) = token_feature_outputs
        else:
            h_E, E_idx, V, Y_nodes, Y_edges, Y_m, D_neighbors = (
                token_feature_outputs
            )
        #! (JH) h_E and E_idx are also considering ligand atoms here.
        #! (JH) but h_E and E_idx are masked out for padded tokens (token_exists_mask is 0 for padded tokens)

        # Prepare protein residue node mask
        protein_residue_node_mask = batch["protein_residue_node_mask"]
        protein_residue_node_mask_2d = gather_nodes(protein_residue_node_mask.unsqueeze(-1), E_idx).squeeze(-1)
        protein_residue_node_mask_2d = protein_residue_node_mask.unsqueeze(-1) * protein_residue_node_mask_2d

        if self.use_shared_atom_triangle:
            if context_metadata is None:
                raise RuntimeError(
                    "shared_atom_triangle requires context atom metadata"
                )
            shared_atom_matches, shared_atom_edge_mask = (
                build_shared_atom_lookup(
                    context_metadata["context_atom_idx"],
                    Y_m,
                    E_idx,
                    protein_residue_node_mask_2d,
                )
            )
        if self.use_mixing_encoder_pair_triangle:
            protein_pair_lookup = build_protein_pair_edge_slot_matches(
                E_idx,
                protein_residue_node_mask_2d,
            )

        # Pass through encoder layers
        # Residue-level encoding, for standard AAs in protein chains only
        h_V = h_V + h_S
        h_E = self.W_e(h_E)

        for layer_index, layer in enumerate(self.encoder_layers):
            h_V, h_E = layer(h_V, h_E, E_idx, protein_residue_node_mask, protein_residue_node_mask_2d)

        # Process ligand context features
        h_V_C_skip = None
        h_E_context_for_shared_atom_triangle = None
        if self.ligand_conditioning:
            context_outputs = self.context_module(
                h_V=h_V,
                h_E=h_E,
                V=V,
                Y_nodes=Y_nodes,
                Y_edges=Y_edges,
                Y_m=Y_m,
                E_idx=E_idx,
                protein_residue_node_mask=protein_residue_node_mask,
                return_context_edges=self.use_shared_atom_triangle,
            )
            if self.use_shared_atom_triangle:
                (
                    h_V,
                    h_V_C_skip,
                    h_E_context_for_shared_atom_triangle,
                ) = context_outputs
            else:
                h_V, h_V_C_skip = context_outputs

        # Add sequence information to the decoder using the selected input contract.
        if self.use_mpnn_decoder:
            if self.use_caliby_decoder:
                h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
                h_E = cat_neighbors_nodes(h_V, h_ES, E_idx)
                for layer in self.decoder_layers:
                    h_V, h_E = layer(
                        h_V=h_V,
                        h_E=h_E,
                        mask_V=protein_residue_node_mask,
                        E_idx=E_idx,
                        mask_attend=protein_residue_node_mask_2d,
                    )
            else:
                h_V = h_V + h_S
                for layer_index, layer in enumerate(self.decoder_layers):
                    h_V, h_E = layer(h_V = h_V, h_E = h_E,
                                        mask_V = protein_residue_node_mask, E_idx = E_idx,
                                        mask_attend = protein_residue_node_mask_2d, h_V_C_skip=h_V_C_skip)
                    if (
                        layer_index
                        in self._shared_atom_triangle_slot_by_mixing_layer
                    ):
                        if (
                            h_E_context_for_shared_atom_triangle is None
                        ):
                            raise RuntimeError(
                                "shared_atom_triangle requires updated "
                                "protein-context edges"
                            )
                        h_E = self._apply_shared_atom_triangle_after_decoder_layer(
                            layer_index=layer_index,
                            protein_context_pairs=(
                                h_E_context_for_shared_atom_triangle
                            ),
                            protein_pairs=h_E,
                            neighbor_idx=E_idx,
                            shared_atom_matches=shared_atom_matches,
                            shared_atom_edge_mask=shared_atom_edge_mask,
                            protein_pair_mask=(
                                protein_residue_node_mask_2d
                            ),
                        )
                    if (
                        layer_index
                        in self._protein_pair_triangle_slot_by_mixing_layer
                    ):
                        triangle_slot = (
                            self._protein_pair_triangle_slot_by_mixing_layer[
                                layer_index
                            ]
                        )
                        outgoing_layer = self.mixing_encoder_outgoing_triangle_layers[
                            triangle_slot
                        ]
                        incoming_layer = self.mixing_encoder_incoming_triangle_layers[
                            triangle_slot
                        ]
                        h_E = self._apply_protein_pair_triangles(
                            outgoing_layer=outgoing_layer,
                            incoming_layer=incoming_layer,
                            protein_pairs=h_E,
                            protein_pair_lookup=protein_pair_lookup,
                        )

        if self.use_caliby_decoder:
            h_V_potts = h_V
        else:
            h_V_potts = self._expand_potts_nodes(h_V, h_V_C_skip)
            h_E = self._expand_potts_edges(
                h_V,
                h_E,
                E_idx,
                h_V_C_skip,
            )

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

            return_multi_head_stats = (
                self.full_multi_head_aggregation is not None and not is_sampling
            )
            potts_output = self.decoder_S_potts(
                h_V_potts,
                h_E,
                E_idx,
                protein_residue_node_mask,
                protein_residue_node_mask_2d,
                return_multi_head_stats=return_multi_head_stats,
            )
            if return_multi_head_stats:
                h, J, multi_head_stats = potts_output
            else:
                h, J = potts_output
                multi_head_stats = None
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
            if multi_head_stats is not None:
                potts_decoder_aux["multi_head_stats"] = multi_head_stats

        logits = self.W_out(h_V)

        # Output features
        mpnn_feature_dict = {"h_V": h_V, "h_ESV": h_E, "E_idx": E_idx} # Todo: Need to change "h_ESV" to "h_E" in the pipeline later
        if self.use_potts:
            mpnn_feature_dict["potts_decoder_aux"] = potts_decoder_aux
        if sidechain_prediction_aux is not None:
            mpnn_feature_dict["sidechain_prediction_aux"] = sidechain_prediction_aux

        return logits, mpnn_feature_dict

    def _expand_potts_nodes(self, h_V, h_V_C_skip=None):
        if not self.use_shared_edge_multi_head_potts:
            return h_V
        if h_V_C_skip is None:
            raise RuntimeError(
                f"expansion_mode={self.expansion_mode!r} requires context "
                "features from ContextModule"
            )
        return torch.cat([h_V, h_V_C_skip], -1)

    def _expand_potts_edges(
        self,
        h_V,
        h_E,
        E_idx,
        h_V_C_skip=None,
    ):
        if self.use_shared_edge_multi_head_potts:
            return self._append_potts_node_context_edges(
                h_V=h_V,
                h_E=h_E,
                h_V_C=h_V_C_skip,
                E_idx=E_idx,
            )

        h_E_neighbors = cat_neighbors_nodes(h_V, h_E, E_idx) # [h_E_ij, h_V_j]
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E_neighbors.size(-2), -1)
        return torch.cat([h_V_expand, h_E_neighbors], -1)

    @staticmethod
    def _append_potts_node_context_edges(h_V, h_E, h_V_C, E_idx):
        if h_V_C is None:
            raise RuntimeError(
                "edge-first Potts expansion requires context features from "
                "ContextModule"
            )
        h_V_j = gather_nodes(h_V, E_idx)
        h_V_i = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_V_C_j = gather_nodes(h_V_C, E_idx)
        h_V_C_i = h_V_C.unsqueeze(-2).expand(
            -1, -1, h_E.size(-2), -1
        )
        return torch.cat([h_E, h_V_i, h_V_j, h_V_C_i, h_V_C_j], -1)
