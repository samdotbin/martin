"""
model.py — GNN encoder + policy/value/aux heads (§6).

Pipeline:
  [B, 64 bars, 28 edges, F] + [positions, exposures, regime, time features]
    -> temporal causal conv over 64 bars (64 -> 1)
    -> GNN over the complete graph of 8 currencies (2-3 message-passing layers)
    -> {global embedding, per-edge embeddings}
    -> policy heads (28, masked-softmax over 5 buckets), value head, aux head

bars_remaining_norm / risk_budget_used are per-episode scalars broadcast
across ALL 28 per-pair heads AND the value head (§3.4, §21 implementation
hint) — easy to silently only reach the value head while everything still
"runs".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class TemporalEncoder(nn.Module):
    """Dilated causal Conv1D stack over the 64-bar lookback, per pair."""

    def __init__(self, in_features, channels=cfg.TEMPORAL_CHANNELS, n_layers=3):
        super().__init__()
        layers = []
        c_in = in_features
        for i in range(n_layers):
            dilation = 2 ** i
            pad = dilation * 2  # kernel_size=3 causal padding
            layers.append(
                nn.Conv1d(c_in, channels, kernel_size=3, dilation=dilation, padding=pad)
            )
            layers.append(nn.ReLU())
            c_in = channels
        self.net = nn.ModuleList(layers)
        self.out_dim = channels

    def forward(self, x):
        # x: [B * n_pairs, F, T]
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.Conv1d) and layer.padding[0] > 0:
                x = x[..., : -layer.padding[0]]  # keep it causal, trim future leakage
        return x[..., -1]  # last timestep -> [B*n_pairs, channels]


class GNNLayer(nn.Module):
    """One message-passing round over the complete graph of currencies."""

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.edge_update = nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim),
        )
        self.node_update = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(self, node_feats, edge_feats, pair_currency_map):
        # node_feats: [B, n_currencies, node_dim], edge_feats: [B, n_pairs, edge_dim]
        B, n_curr, node_dim = node_feats.shape
        n_pairs = edge_feats.shape[1]

        new_edges = []
        for p, (b_idx, q_idx) in enumerate(pair_currency_map):
            msg_in = torch.cat([edge_feats[:, p], node_feats[:, b_idx], node_feats[:, q_idx]], dim=-1)
            new_edges.append(self.edge_update(msg_in))
        new_edge_feats = torch.stack(new_edges, dim=1)  # [B, n_pairs, edge_dim]

        agg = torch.zeros(B, n_curr, new_edge_feats.shape[-1], device=node_feats.device)
        counts = torch.zeros(B, n_curr, 1, device=node_feats.device) + 1e-9
        for p, (b_idx, q_idx) in enumerate(pair_currency_map):
            agg[:, b_idx] += new_edge_feats[:, p]
            agg[:, q_idx] += new_edge_feats[:, p]
            counts[:, b_idx] += 1
            counts[:, q_idx] += 1
        agg = agg / counts

        new_nodes = self.node_update(torch.cat([node_feats, agg], dim=-1))
        return node_feats + new_nodes, edge_feats + new_edge_feats  # residual


class CurrencyGNN(nn.Module):
    def __init__(self, pair_currency_map, n_currencies=8, n_pairs=28,
                 edge_in=cfg.EDGE_FEATURES, hidden=cfg.GNN_HIDDEN_DIM,
                 n_layers=cfg.GNN_MESSAGE_PASSING_LAYERS, extra_scalar_dim=0):
        super().__init__()
        self.pair_currency_map = pair_currency_map
        self.n_currencies = n_currencies
        self.n_pairs = n_pairs

        self.temporal = TemporalEncoder(edge_in)
        temporal_out = self.temporal.out_dim

        # extra per-episode scalars/side info appended to every edge embedding
        # before message passing: positions, regime embedding, time features.
        self.edge_proj = nn.Linear(temporal_out + 1 + extra_scalar_dim, hidden)  # +1 = own position
        self.node_embed = nn.Parameter(torch.randn(n_currencies, hidden) * 0.01)

        self.gnn_layers = nn.ModuleList(
            [GNNLayer(hidden, hidden, hidden) for _ in range(n_layers)]
        )
        self.out_dim = hidden

    def forward(self, edge_history, positions, side_info):
        """
        edge_history: [B, 64, n_pairs, F]
        positions:    [B, n_pairs]
        side_info:    [B, side_dim]  (regime embedding + time features etc.), broadcast to edges
        returns: global_embedding [B, hidden], per_edge_embeddings [B, n_pairs, hidden]
        """
        B, T, P, Fdim = edge_history.shape
        x = edge_history.permute(0, 2, 3, 1).reshape(B * P, Fdim, T)  # [B*P, F, T]
        temporal_out = self.temporal(x).reshape(B, P, -1)  # [B, P, temporal_out]

        side_broadcast = side_info.unsqueeze(1).expand(-1, P, -1)  # [B, P, side_dim]
        edge_in = torch.cat([temporal_out, positions.unsqueeze(-1), side_broadcast], dim=-1)
        edge_feats = F.relu(self.edge_proj(edge_in))  # [B, P, hidden]

        node_feats = self.node_embed.unsqueeze(0).expand(B, -1, -1).contiguous()

        for layer in self.gnn_layers:
            node_feats, edge_feats = layer(node_feats, edge_feats, self.pair_currency_map)

        global_embedding = torch.cat(
            [node_feats.mean(dim=1), edge_feats.mean(dim=1)], dim=-1
        )
        return global_embedding, edge_feats


class PolicyValueNet(nn.Module):
    def __init__(self, pair_currency_map, n_buckets=len(cfg.ACTION_BUCKETS),
                 regime_dim=cfg.N_REGIME_COMPONENTS + cfg.N_REGIME_CLUSTERS,
                 time_feat_dim=11, n_currencies=8, n_pairs=28):
        # time_feat_dim=11 must match rollout.build_time_features(): 2 (sin/cos)
        # + 5 (day-of-week one-hot) + 4 (session flag one-hot). Keep these in
        # sync — a mismatch fails fast at the first forward() call.
        super().__init__()
        side_dim = regime_dim + time_feat_dim + 2  # +2 = bars_remaining_norm, risk_budget_used
        self.encoder = CurrencyGNN(
            pair_currency_map, n_currencies=n_currencies, n_pairs=n_pairs,
            extra_scalar_dim=side_dim,
        )
        hidden = self.encoder.out_dim
        global_dim = hidden * 2  # node-mean concat edge-mean

        head_in = hidden + global_dim + regime_dim + 2  # edge_emb + global + regime + time-scalars
        self.policy_head = nn.Sequential(
            nn.Linear(head_in, cfg.POLICY_HEAD_HIDDEN), nn.ReLU(),
            nn.Linear(cfg.POLICY_HEAD_HIDDEN, n_buckets),
        )
        value_in = global_dim + hidden + regime_dim + 2
        self.value_head = nn.Sequential(
            nn.Linear(value_in, cfg.POLICY_HEAD_HIDDEN), nn.ReLU(),
            nn.Linear(cfg.POLICY_HEAD_HIDDEN, 1),
        )
        self.aux_head = nn.Sequential(
            nn.Linear(hidden, cfg.POLICY_HEAD_HIDDEN), nn.ReLU(),
            nn.Linear(cfg.POLICY_HEAD_HIDDEN, 1),
        )
        self.n_pairs = n_pairs
        self.n_buckets = n_buckets

    def forward(self, edge_history, positions, regime_embedding, time_features,
                bars_remaining_norm, risk_budget_used, action_mask=None):
        """
        bars_remaining_norm, risk_budget_used: [B] scalars, broadcast to every
        per-pair policy head AND the value head (§3.4) — not just the value head.
        """
        time_scalars = torch.stack([bars_remaining_norm, risk_budget_used], dim=-1)  # [B, 2]
        side_info = torch.cat([regime_embedding, time_features, time_scalars], dim=-1)

        global_embedding, edge_feats = self.encoder(edge_history, positions, side_info)
        B, P, hidden = edge_feats.shape

        global_bcast = global_embedding.unsqueeze(1).expand(-1, P, -1)
        regime_bcast = regime_embedding.unsqueeze(1).expand(-1, P, -1)
        time_bcast = time_scalars.unsqueeze(1).expand(-1, P, -1)
        policy_in = torch.cat([edge_feats, global_bcast, regime_bcast, time_bcast], dim=-1)
        logits = self.policy_head(policy_in)  # [B, P, n_buckets]

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))
        probs = F.softmax(logits, dim=-1)

        value_in = torch.cat(
            [global_embedding, edge_feats.mean(dim=1), regime_embedding, time_scalars], dim=-1
        )
        value = self.value_head(value_in).squeeze(-1)  # [B]

        aux_pred = self.aux_head(edge_feats).squeeze(-1)  # [B, P] predicted next-bar return * 100

        return {"logits": logits, "probs": probs, "value": value, "aux_pred": aux_pred}
