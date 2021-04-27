import numpy as np
import torch

from settings import COURT_LENGTH, COURT_WIDTH, GAMES_DIR
from torch.utils.data import Dataset

# The raw data is recorded at 25 Hz.
RAW_DATA_HZ = 25


class Baller2VecPlusPlusDataset(Dataset):
    def __init__(
        self,
        hz,
        secs,
        N,
        player_traj_n,
        max_player_move,
        gameids,
        starts,
        mode,
        n_player_ids,
    ):
        self.skip = int(RAW_DATA_HZ / hz)
        self.chunk_size = int(RAW_DATA_HZ * secs)
        self.seq_len = self.chunk_size // self.skip

        self.N = N
        self.n_players = 10
        self.gameids = gameids
        self.starts = starts
        self.mode = mode
        self.n_player_ids = n_player_ids

        self.player_traj_n = player_traj_n
        self.max_player_move = max_player_move
        self.player_traj_bins = np.linspace(
            -max_player_move, max_player_move, player_traj_n - 1
        )

    def __len__(self):
        return self.N

    def get_sample(self, X, start):
        # Downsample.
        seq_data = X[start : start + self.chunk_size : self.skip]

        keep_players = np.random.choice(np.arange(10), self.n_players, False)
        if self.mode in {"valid", "test"}:
            keep_players.sort()

        # End sequence early if there is a position glitch. Often happens when there was
        # a break in the game, but glitches also happen for other reasons. See
        # glitch_example.py for an example.
        player_xs = seq_data[:, 20:30][:, keep_players]
        player_ys = seq_data[:, 30:40][:, keep_players]
        player_x_diffs = np.diff(player_xs, axis=0)
        player_y_diffs = np.diff(player_ys, axis=0)
        try:
            glitch_x_break = np.where(
                np.abs(player_x_diffs) > 1.2 * self.max_player_move
            )[0].min()
        except ValueError:
            glitch_x_break = len(seq_data)

        try:
            glitch_y_break = np.where(
                np.abs(player_y_diffs) > 1.2 * self.max_player_move
            )[0].min()
        except ValueError:
            glitch_y_break = len(seq_data)

        seq_break = min(glitch_x_break, glitch_y_break)
        seq_data = seq_data[:seq_break]

        player_idxs = seq_data[:, 10:20][:, keep_players].astype(int)
        player_xs = seq_data[:, 20:30][:, keep_players]
        player_ys = seq_data[:, 30:40][:, keep_players]
        player_hoop_sides = seq_data[:, 40:50][:, keep_players].astype(int)

        # Randomly rotate the court because the hoop direction is arbitrary.
        if (self.mode == "train") and (np.random.random() < 0.5):
            player_xs = COURT_LENGTH - player_xs
            player_ys = COURT_WIDTH - player_ys
            player_hoop_sides = (player_hoop_sides + 1) % 2

        # Get player trajectories.
        player_x_diffs = np.diff(player_xs, axis=0)
        player_y_diffs = np.diff(player_ys, axis=0)

        player_traj_rows = np.digitize(player_y_diffs, self.player_traj_bins)
        player_traj_cols = np.digitize(player_x_diffs, self.player_traj_bins)
        player_trajs = player_traj_rows * self.player_traj_n + player_traj_cols

        return {
            "player_idxs": torch.LongTensor(player_idxs[: seq_break - 1]),
            "player_xs": torch.Tensor(player_xs[: seq_break - 1]),
            "player_ys": torch.Tensor(player_ys[: seq_break - 1]),
            "player_x_diffs": torch.Tensor(player_x_diffs),
            "player_y_diffs": torch.Tensor(player_y_diffs),
            "player_hoop_sides": torch.Tensor(player_hoop_sides[: seq_break - 1]),
            "player_trajs": torch.LongTensor(player_trajs),
        }

    def __getitem__(self, idx):
        if self.mode == "train":
            gameid = np.random.choice(self.gameids)

        elif self.mode in {"valid", "test"}:
            gameid = self.gameids[idx]

        X = np.load(f"{GAMES_DIR}/{gameid}_X.npy")

        if self.mode == "train":
            start = np.random.randint(len(X) - self.chunk_size)

        elif self.mode in {"valid", "test"}:
            start = self.starts[idx]

        return self.get_sample(X, start)
