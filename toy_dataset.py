import numpy as np
import torch

from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(self):
        self.seq_len = 20
        self.n_players = 2
        self.n_player_ids = 2
        player_0_idxs = np.array(self.seq_len * [0])
        player_1_idxs = np.array(self.seq_len * [1])
        self.player_idxs = np.vstack([player_0_idxs, player_1_idxs]).T
        self.player_hoop_sides = np.zeros((self.seq_len, 2))
        self.player_traj_n = 3

    def __len__(self):
        return 500

    def get_sample(self):
        # Sample coordinated trajectories.
        player_x_diffs = np.random.randint(-1, 2, self.seq_len)
        player_y_diffs = np.random.randint(-1, 2, self.seq_len)

        # Calculate total distance traveled in x and y directions.
        player_x_dists = np.cumsum(player_x_diffs)[None].T
        player_y_dists = np.cumsum(player_y_diffs)[None].T

        # Calculate x and y positions for agents.
        player_xs = np.zeros((self.seq_len, self.n_players))
        # Agents start one unit to the left and right of the origin.
        player_xs[0] = [-1, 1]
        player_xs[1:] = player_xs[0] + player_x_dists[: self.seq_len - 1]
        player_ys = np.zeros((self.seq_len, self.n_players))
        player_ys[1:] = player_y_dists[: self.seq_len - 1]

        # Agents start in random order from left to right.
        keep_players = np.random.choice(np.arange(2), self.n_players, False)
        player_idxs = self.player_idxs[:, keep_players].astype(int)

        # Convert trajectories to index labels.
        player_traj_rows = 1 + player_y_diffs
        player_traj_cols = 1 + player_x_diffs
        player_trajs = player_traj_rows * self.player_traj_n + player_traj_cols
        player_trajs = np.vstack([player_trajs, player_trajs]).T

        player_x_diffs = np.vstack([player_x_diffs, player_x_diffs]).T
        player_y_diffs = np.vstack([player_y_diffs, player_y_diffs]).T

        return {
            "player_idxs": torch.LongTensor(player_idxs),
            "player_xs": torch.Tensor(player_xs),
            "player_ys": torch.Tensor(player_ys),
            "player_x_diffs": torch.Tensor(player_x_diffs),
            "player_y_diffs": torch.Tensor(player_y_diffs),
            "player_hoop_sides": torch.Tensor(self.player_hoop_sides),
            "player_trajs": torch.LongTensor(player_trajs),
        }

    def __getitem__(self, idx):
        return self.get_sample()
