import math
import torch

from torch import nn


class Baller2VecPlusPlus(nn.Module):
    def __init__(
        self,
        n_player_ids,
        embedding_dim,
        sigmoid,
        seq_len,
        mlp_layers,
        n_players,
        n_player_labels,
        nhead,
        dim_feedforward,
        num_layers,
        dropout,
        b2v,
    ):
        super().__init__()
        self.sigmoid = sigmoid
        self.seq_len = seq_len
        self.n_players = n_players
        self.b2v = b2v

        initrange = 0.1
        self.player_embedding = nn.Embedding(n_player_ids, embedding_dim)
        self.player_embedding.weight.data.uniform_(-initrange, initrange)

        start_mlp = nn.Sequential()
        pos_mlp = nn.Sequential()
        traj_mlp = nn.Sequential()
        pos_in_feats = embedding_dim + 3
        traj_in_feats = embedding_dim + 5
        for (layer_idx, out_feats) in enumerate(mlp_layers):
            start_mlp.add_module(
                f"layer{layer_idx}", nn.Linear(pos_in_feats, out_feats)
            )
            pos_mlp.add_module(f"layer{layer_idx}", nn.Linear(pos_in_feats, out_feats))
            traj_mlp.add_module(
                f"layer{layer_idx}", nn.Linear(traj_in_feats, out_feats)
            )
            if layer_idx < len(mlp_layers) - 1:
                start_mlp.add_module(f"relu{layer_idx}", nn.ReLU())
                pos_mlp.add_module(f"relu{layer_idx}", nn.ReLU())
                traj_mlp.add_module(f"relu{layer_idx}", nn.ReLU())

            pos_in_feats = out_feats
            traj_in_feats = out_feats

        self.start_mlp = start_mlp
        self.pos_mlp = pos_mlp
        self.traj_mlp = traj_mlp

        d_model = mlp_layers[-1]
        self.d_model = d_model
        if b2v:
            self.register_buffer("b2v_mask", self.generate_b2v_mask())
        else:
            self.register_buffer("b2vpp_mask", self.generate_b2vpp_mask())

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.traj_classifier = nn.Linear(d_model, n_player_labels)
        self.traj_classifier.weight.data.uniform_(-initrange, initrange)
        self.traj_classifier.bias.data.zero_()

    def generate_b2v_mask(self):
        seq_len = self.seq_len
        n_players = self.n_players
        sz = seq_len * n_players
        mask = torch.zeros(sz, sz)
        for step in range(seq_len):
            start = step * n_players
            stop = start + n_players
            mask[start:stop, :stop] = 1

        mask = mask.masked_fill(mask == 0, float("-inf"))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def generate_b2vpp_mask(self):
        n_players = self.n_players
        tri_sz = 2 * (self.seq_len * n_players)
        sz = tri_sz + n_players
        mask = torch.zeros(sz, sz)
        mask[:tri_sz, :tri_sz] = torch.tril(torch.ones(tri_sz, tri_sz))
        mask[:, -n_players:] = 1
        mask = mask.masked_fill(mask == 0, float("-inf"))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tensors):
        device = list(self.player_embedding.parameters())[0].device

        player_embeddings = self.player_embedding(
            tensors["player_idxs"].flatten().to(device)
        )
        if self.sigmoid == "logistic":
            player_embeddings = torch.sigmoid(player_embeddings)
        elif self.sigmoid == "tanh":
            player_embeddings = torch.tanh(player_embeddings)

        player_xs = tensors["player_xs"].flatten().unsqueeze(1).to(device)
        player_ys = tensors["player_ys"].flatten().unsqueeze(1).to(device)
        player_hoop_sides = (
            tensors["player_hoop_sides"].flatten().unsqueeze(1).to(device)
        )
        player_pos = torch.cat(
            [
                player_embeddings,
                player_xs,
                player_ys,
                player_hoop_sides,
            ],
            dim=1,
        )
        pos_feats = self.pos_mlp(player_pos) * math.sqrt(self.d_model)
        if self.b2v:
            outputs = self.transformer(pos_feats.unsqueeze(1), self.b2v_mask)
            preds = self.traj_classifier(outputs.squeeze(1))

        else:
            start_feats = self.start_mlp(player_pos[: self.n_players]) * math.sqrt(
                self.d_model
            )

            player_x_diffs = tensors["player_x_diffs"].flatten().unsqueeze(1).to(device)
            player_y_diffs = tensors["player_y_diffs"].flatten().unsqueeze(1).to(device)
            player_trajs = torch.cat(
                [
                    player_embeddings,
                    player_xs + player_x_diffs,
                    player_ys + player_y_diffs,
                    player_hoop_sides,
                    player_x_diffs,
                    player_y_diffs,
                ],
                dim=1,
            )
            trajs_feats = self.traj_mlp(player_trajs) * math.sqrt(self.d_model)

            combined = torch.zeros(
                2 * len(pos_feats) + self.n_players, self.d_model
            ).to(device)
            combined[: -self.n_players : 2] = pos_feats
            combined[1 : -self.n_players : 2] = trajs_feats
            combined[-self.n_players :] = start_feats

            outputs = self.transformer(combined.unsqueeze(1), self.b2vpp_mask)
            preds = self.traj_classifier(outputs.squeeze(1)[: -self.n_players : 2])

        return preds
