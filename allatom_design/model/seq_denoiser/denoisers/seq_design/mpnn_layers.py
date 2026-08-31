"""Protein, context-conditioning, and mixing-encoder layers for ElixMPNN."""

import torch
import torch.nn as nn

from allatom_design.model.seq_denoiser.denoisers.seq_design.mpnn_utils import (
    factorized_triplet_linear,
    gather_nodes,
)


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


class ProteinEncoderLayer(nn.Module):
    """Update protein nodes and, except at the configured end, protein edges."""

    def __init__(
        self,
        num_hidden,
        num_in,
        dropout=0.1,
        scale=30,
        is_last_layer=False,
    ):
        super().__init__()
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

        h_V_j = gather_nodes(h_V, E_idx)
        first_hidden = factorized_triplet_linear(
            self.W1,
            h_V.unsqueeze(-2),
            h_E,
            h_V_j,
        )
        h_message = self.W3(
            self.act(self.W2(self.act(first_hidden)))
        ) # [B, L, K, num_hidden]

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
            h_V_j = gather_nodes(h_V, E_idx)
            first_hidden = factorized_triplet_linear(
                self.W11,
                h_V.unsqueeze(-2),
                h_E,
                h_V_j,
            )
            h_message = self.W13(
                self.act(self.W12(self.act(first_hidden)))
            ) # [B, L, K, num_hidden]
            h_E = self.norm3(h_E + self.dropout3(h_message))
            if mask_attend is not None:
                h_E = mask_attend.unsqueeze(-1) * h_E

        return h_V, h_E


class MixingEncoderLayer(nn.Module):
    """Mix sequence-conditioned protein nodes, edges, and optional context."""

    def __init__(
        self,
        num_hidden,
        num_in,
        dropout=0.1,
        scale=30,
        is_last_layer=False,
        use_context_skip_connection=False,
    ):
        super().__init__()
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

        h_V_j = gather_nodes(h_V, E_idx)
        first_hidden = factorized_triplet_linear(
            self.W1,
            h_V.unsqueeze(-2),
            h_E,
            h_V_j,
        )
        h_message = self.W3(
            self.act(self.W2(self.act(first_hidden)))
        ) # [B, L, K, num_hidden]

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
        h_V_j = gather_nodes(h_V, E_idx)
        first_hidden = factorized_triplet_linear(
            self.W11,
            h_V.unsqueeze(-2),
            h_E,
            h_V_j,
        )
        h_message = self.W13(
            self.act(self.W12(self.act(first_hidden)))
        ) # [B, L, K, num_hidden]
        h_E = self.norm3(h_E + self.dropout3(h_message))
        if mask_attend is not None:
            h_E = mask_attend.unsqueeze(-1) * h_E

        return h_V, h_E


class CalibyMixingEncoderLayer(nn.Module):
    """Update H-wide nodes and the persistent 3H Caliby mixing edge state."""

    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_in)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden * 2 + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_in, bias=True)

        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message

        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1) * h_V

        h_V_j = gather_nodes(h_V, E_idx)
        first_hidden = factorized_triplet_linear(
            self.W11,
            h_V.unsqueeze(-2),
            h_E,
            h_V_j,
        )
        h_message = self.W13(
            self.act(self.W12(self.act(first_hidden)))
        )
        h_E = self.norm3(h_E + self.dropout3(h_message))
        if mask_attend is not None:
            h_E = mask_attend.unsqueeze(-1) * h_E

        return h_V, h_E


class ContextEncoderLayer(nn.Module):
    """Update context nodes and optionally context-context edges."""

    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None,
                 scale=30, update_edges=False):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.update_edges = update_edges
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.num_hidden)
        self.norm2 = nn.LayerNorm(self.num_hidden)

        # Node update
        self.W1 = nn.Linear(self.num_in, self.num_hidden, bias=True) # Following the foundry's LigandMPNN implementation
        self.W2 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)
        self.W3 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)

        # Edge update
        if self.update_edges:
            self.W11 = nn.Linear(self.num_in, self.num_hidden, bias=True) # nh * 2 for vi AND vj
            self.W12 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)
            self.W13 = nn.Linear(self.num_hidden, self.num_hidden, bias=True) # num_in is hidden dim of edges h_E
            self.norm3 = nn.LayerNorm(self.num_hidden)
            self.dropout3 = nn.Dropout(dropout)

        # Activation and feedforward
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(self.num_hidden, num_hidden * 4)

    def forward(self, h_V = None, h_E = None, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        first_hidden = factorized_triplet_linear(
            self.W1,
            h_V.unsqueeze(-2),
            h_E,
            h_V.unsqueeze(-3),
        )
        h_message = self.W3(self.act(self.W2(self.act(first_hidden))))

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

        if self.update_edges:
            first_hidden = factorized_triplet_linear(
                self.W11,
                h_V.unsqueeze(-2),
                h_E,
                h_V.unsqueeze(-3),
            )
            h_message = self.W13(
                self.act(self.W12(self.act(first_hidden)))
            )
            h_E = self.norm3(h_E + self.dropout3(h_message))

            if mask_attend is not None:
                h_E = mask_attend.unsqueeze(-1) * h_E

            return h_V, h_E
        else:
            return h_V, None


class ContextToProteinLayer(nn.Module):
    """Aggregate context nodes into protein nodes and protein-context edges."""

    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None,
                 scale=30, update_edges=False):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.update_edges = update_edges

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.num_hidden)
        self.norm2 = nn.LayerNorm(self.num_hidden)

        # Node update
        self.W1 = nn.Linear(self.num_in, self.num_hidden, bias=True)
        self.W2 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)
        self.W3 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)

        # Edge update
        if self.update_edges:
            self.W11 = nn.Linear(self.num_in, self.num_hidden, bias=True)
            self.W12 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)
            self.W13 = nn.Linear(self.num_hidden, self.num_hidden, bias=True) # num_in is hidden dim of edges h_E
            self.norm3 = nn.LayerNorm(self.num_hidden)
            self.dropout3 = nn.Dropout(dropout)

        # Activation and feedforward
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(self.num_hidden, num_hidden * 4)

    def forward(self, h_V = None, h_E_context = None, Y_nodes = None, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        first_hidden = factorized_triplet_linear(
            self.W1,
            h_V.unsqueeze(-2),
            h_E_context,
            Y_nodes,
        )
        h_message = self.W3(self.act(self.W2(self.act(first_hidden))))
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
        if self.update_edges:
            first_hidden = factorized_triplet_linear(
                self.W11,
                h_V.unsqueeze(-2),
                h_E_context,
                Y_nodes,
            )
            h_message = self.W13(
                self.act(self.W12(self.act(first_hidden)))
            )
            h_E_context = self.norm3(h_E_context + self.dropout3(h_message))

            if mask_attend is not None:
                h_E_context = mask_attend.unsqueeze(-1) * h_E_context

            return h_V, h_E_context
        else:
            return h_V, None


class ContextConditioner(nn.Module):
    """Encode the context graph, then condition protein representations."""

    def __init__(
        self,
        hidden_dim: int,
        dropout_p: float,
        num_context_encoder_layers: int,
        num_context_to_protein_layers: int,
        update_context_edges: bool,
        update_protein_context_edges: bool,
        return_context_skip: bool,
        update_final_context_edge: bool = True,
        update_final_protein_context_edge: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.update_context_edges = update_context_edges
        self.update_protein_context_edges = update_protein_context_edges
        self.update_final_context_edge = update_final_context_edge
        self.update_final_protein_context_edge = update_final_protein_context_edge
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
        self.context_encoder_layers = torch.nn.ModuleList(
            [ContextEncoderLayer(self.hidden_dim, 3 * self.hidden_dim,
                                 dropout=dropout_p,
                                 update_edges=(
                                     self.update_context_edges
                                     and (
                                         self.update_final_context_edge
                                         or i < num_context_encoder_layers - 1
                                     )
                                 )) for i in range(num_context_encoder_layers)]

        )

        self.context_to_protein_layers = torch.nn.ModuleList(
            [ContextToProteinLayer(self.hidden_dim, 3 * self.hidden_dim,
                                   dropout=dropout_p,
                                   update_edges=(
                                       self.update_protein_context_edges
                                       and (
                                           self.update_final_protein_context_edge
                                           or i < num_context_to_protein_layers - 1
                                       )
                                   )) for i in range(num_context_to_protein_layers)]
        )

    def forward(self, h_V = None, h_E = None,
                V = None, Y_nodes = None, Y_edges = None,
                Y_m = None, E_idx = None,
                protein_residue_node_mask = None,
                return_context_edges: bool = False):
        h_E_context = self.W_v(V)
        h_V_C = self.W_c(h_V)
        Y_m_edges = Y_m[:, :, :, None] * Y_m[:, :, None, :]
        Y_nodes = self.W_nodes_y(Y_nodes)
        Y_edges = self.W_edges_y(Y_edges)

        for layer in self.context_encoder_layers:
            Y_nodes, updated_Y_edges = layer(
                h_V=Y_nodes,
                h_E=Y_edges,
                mask_V=Y_m,
                mask_attend=Y_m_edges,
            )
            if updated_Y_edges is not None:
                Y_edges = updated_Y_edges

        for layer in self.context_to_protein_layers:
            h_V_C, updated_h_E_context = layer(
                h_V=h_V_C,
                h_E_context=h_E_context,
                Y_nodes=Y_nodes,
                mask_V=protein_residue_node_mask,
                mask_attend=Y_m,
            )
            if updated_h_E_context is not None:
                h_E_context = updated_h_E_context

        h_V_C_skip = h_V_C if self.return_context_skip else None

        h_V_C = self.V_C(h_V_C)
        h_V = h_V + self.V_C_norm(self.dropout(h_V_C))
        if return_context_edges:
            return h_V, h_V_C_skip, h_E_context
        return h_V, h_V_C_skip
