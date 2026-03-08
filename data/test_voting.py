# test_voting.py

import numpy as np
import torch
from collections import defaultdict

from typing import Callable, List, Optional, Tuple, Dict

class RoomVoter:
    """Accumulates per-point predictions across overlapping crops."""

    def __init__(self, num_classes: int = 13):
        self.num_classes = num_classes
        # room_id -> (N, num_classes) vote accumulator
        self.votes: Dict[int, np.ndarray] = {}
        self.counts: Dict[int, np.ndarray] = {}

    def update(self, room_id: int, point_indices: np.ndarray,
               logits: np.ndarray):
        """
        room_id: int
        point_indices: (M,) original indices in full room
        logits: (M, num_classes)
        """
        rid = int(room_id)
        if rid not in self.votes:
            # lazy init
            max_idx = point_indices.max() + 1
            self.votes[rid] = np.zeros((max_idx, self.num_classes), np.float32)
            self.counts[rid] = np.zeros(max_idx, np.int32)

        self.votes[rid][point_indices] += logits
        self.counts[rid][point_indices] += 1

    def get_predictions(self, room_id: int) -> np.ndarray:
        rid = int(room_id)
        votes = self.votes[rid]
        counts = self.counts[rid].clip(min=1)[:, None]
        return (votes / counts).argmax(axis=1)