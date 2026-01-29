from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from typing import List, Tuple


class TeleopDataset(Dataset):
    STATE_COLS = ["puppet_x", "puppet_y", "puppet_z",
                  "goal_x", "goal_y", "goal_z"]
    ACTION_COLS = ["master_x", "master_y", "master_z"]

    def __init__(self, data_path: str):
        # Scan for .csv files
        data_paths = sorted(Path(data_path).glob("*.csv"))
        if len(data_paths) == 0:
            raise FileNotFoundError(f"No .csv files found under: {data_path}")

        # Store per-trajectory arrays
        self.traj_states: List[np.ndarray] = []
        self.traj_actions: List[np.ndarray] = []
        self.traj_lengths: List[int] = []

        # Build a global index map of valid
        self.index: List[Tuple[int, int]] = []

        for traj_id, path in enumerate(data_paths):
            states, actions = self._load_one_csv(path)

            assert states.shape[0] == actions.shape[0] + 1, \
                f"Trajectory length mismatch in {path.name}: " \
                f"states {states.shape[0]} vs actions {actions.shape[0]}"

            T = int(actions.shape[0])

            # Need at least 2 steps to form (s_t, a_t, s_{t+1})
            if T < 2:
                continue

            self.traj_states.append(states.astype(np.float32))
            self.traj_actions.append(actions.astype(np.float32))
            self.traj_lengths.append(T)
            self.index.extend([(traj_id, t) for t in range(T)])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        raise NotImplementedError

    def _load_one_csv(self, path: Path):
        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            raise ValueError(f"{path.name} missing required column: timestamp")

        df = df.sort_values("timestamp").reset_index(drop=True)

        is_state = df[self.STATE_COLS].notna().all(axis=1)
        is_action = df[self.ACTION_COLS].notna().all(axis=1)

        df = df[is_state | is_action].reset_index(drop=True)

        if len(df) < 3:
            return (np.zeros((0, 6), np.float32),
                    np.zeros((0, 3), np.float32))

        types = np.where(df[self.STATE_COLS].notna().all(axis=1), 0, 1)

        keep = np.zeros(len(types), dtype=bool)
        expect = 0  # start with state

        for i, t in enumerate(types):
            if t == expect:
                keep[i] = True
                expect = 1 - expect

        df = df[keep].reset_index(drop=True)

        # must end with state
        if df[self.ACTION_COLS].notna().all(axis=1).iloc[-1]:
            df = df.iloc[:-1]

        if len(df) < 3:
            return (np.zeros((0, 6), np.float32),
                    np.zeros((0, 3), np.float32))

        states = df.iloc[0::2][self.STATE_COLS].to_numpy(dtype=np.float32)
        actions = df.iloc[1::2][self.ACTION_COLS].to_numpy(dtype=np.float32)

        return states, actions


class MDPTeleopDataset(TeleopDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        traj_id, t = self.index[idx]

        s = torch.from_numpy(self.traj_states[traj_id][t])          # (6,)
        a = torch.from_numpy(self.traj_actions[traj_id][t])         # (3,)
        s_next = torch.from_numpy(self.traj_states[traj_id][t+1])   # (6,)

        return s, a, s_next


class SeqTeleopDataset(TeleopDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.traj_lengths)

    def __getitem__(self, idx):
        t = self.traj_lengths[idx]

        s_seq = torch.from_numpy(self.traj_states[idx][:t])         # (T, 6)
        a_seq = torch.from_numpy(self.traj_actions[idx][:t])        # (T, 3)
        s_next_seq = torch.from_numpy(self.traj_states[idx][1:t+1]) # (T, 6)

        return s_seq, a_seq, s_next_seq
