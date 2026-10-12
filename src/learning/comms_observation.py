# For learned communication
#
# This turns one robot's state at one tick into fixed-shape numpy tensors the
# comms policy (sender + receive/trust heads) will eventually consume. It reads
# ONLY quantities the comms policy is allowed to see -- the robot's own belief,
# its first-hand sensed_mask, its per-cell comms trust overlay, the messages that
# were actually DELIVERED to it, episode-level comms accounting (CommsStats), and
# the current tick. Ground truth never enters here (invariant: policies see
# observations, not the Map).
#
# It does NOT touch movement: no action is chosen, no world state mutated. It is a
# pure, deterministic feature extractor. No PyTorch / PPO here - outputs are
# plain float32 arrays so the learning stack can wrap them however it likes.
#
# Shapes come from config.py (the single source of truth), so the observation
# space, this builder, and the eventual policy always agree:
#   robot_state       (ROBOT_D,)
#   message_features  (MAX_MESSAGES, MESSAGE_D)
#   message_mask      (MAX_MESSAGES,)                 1.0 real, 0.0 padding
#   resource_features (RESOURCE_D,)                   budget / comms accounting
#   frontier_summary  (FRONTIER_SUMMARY_D,)           optional; zeros if disabled

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from comms.backend import CommsStats
from comms.delivered import DeliveredMessage
from config import (
    FRONTIER_SUMMARY_D,
    MAX_FRONTIERS,
    MAX_MESSAGES,
    MESSAGE_D,
    RESOURCE_D,
    ROBOT_D,
)
from planning.frontiers import cluster_frontiers, find_frontier_cells
from robot import UNKNOWN, Position

# Normalization scales for the unbounded resource counters. These are display
# scales only -- they map raw accumulators into a bounded [0, 1] range so the
# network sees stationary inputs; they carry no behavioral meaning and can be
# retuned without changing the observation's shape or ordering.
DEFAULT_TICK_SCALE = 500.0    # rough episode horizon
DEFAULT_BYTE_SCALE = 100_000.0  # rough per-episode payload-byte budget

# Bytes on the wire per belief cell (row, col, value as three int16s); mirrors
# comms.message._BYTES_PER_CELL, duplicated here to avoid importing a private.
_BYTES_PER_CELL = 6


@dataclass(frozen=True)
class CommsObservation:
    # Fixed-shape, owned, read-only tensors for one robot on one tick.
    robot_state: np.ndarray        # (ROBOT_D,)
    message_features: np.ndarray   # (MAX_MESSAGES, MESSAGE_D)
    message_mask: np.ndarray       # (MAX_MESSAGES,)  1.0 real / 0.0 padding
    resource_features: np.ndarray  # (RESOURCE_D,)
    frontier_summary: np.ndarray   # (FRONTIER_SUMMARY_D,)

    @property
    def num_messages(self) -> int:
        # How many message rows are real (unpadded) this tick.
        return int(self.message_mask.sum())


def _message_sort_key(m: DeliveredMessage) -> tuple[int, str, int]:
    # Deterministic ordering independent of delivery/list order: oldest first,
    # then sender id, then sender-scoped sequence id. Two builds over the same set
    # of delivered messages therefore always produce byte-identical tensors.
    return (m.created_tick, m.sender_id, m.sequence_id)


def _message_row(
    m: DeliveredMessage,
    belief_map: np.ndarray,
    position: Position,
    height: int,
    width: int,
) -> np.ndarray:
    # One (MESSAGE_D,) feature row for a delivered message, scored against the
    # receiver's CURRENT belief. All features are bounded so padding (all-zeros)
    # is a valid, distinguishable "no message" row.
    total_cells = float(height * width)
    n_cells = len(m.cells)

    novel = agree = conflict = 0
    for (r, c), value in m.cells:
        if not (0 <= r < height and 0 <= c < width):
            continue  # out-of-bounds claim contributes to none of the counts
        current = belief_map[r, c]
        if current == UNKNOWN:
            novel += 1
        elif current == value:
            agree += 1
        else:
            conflict += 1
    denom = float(n_cells) if n_cells else 1.0

    age = m.age  # ticks in flight; >= 1 under next-tick delivery
    if m.sender_position is not None:
        sr, sc = m.sender_position
        sender_dist = (abs(sr - position[0]) + abs(sc - position[1])) / float(
            height + width
        )
        has_pos = 1.0
    else:
        sender_dist = 0.0
        has_pos = 0.0

    return np.array(
        [
            age / (age + 1.0),                                   # 0 staleness (smooth, bounded)
            min(n_cells / total_cells, 1.0),                     # 1 payload size (cells)
            min(m.payload_size_bytes / (total_cells * _BYTES_PER_CELL), 1.0),  # 2 payload size (bytes)
            novel / denom,                                       # 3 novel fraction
            agree / denom,                                       # 4 agree-with-belief fraction
            conflict / denom,                                    # 5 conflict-with-belief fraction
            min(sender_dist, 1.0),                               # 6 claimed-sender distance
            has_pos,                                             # 7 sender-position present
        ],
        dtype=np.float32,
    )


def _frontier_summary(
    belief_map: np.ndarray,
    position: Position,
    height: int,
    width: int,
) -> np.ndarray:
    # Compact GLOBAL summary of the belief's frontiers -- not a per-frontier tensor.
    # Cheap to compute from what we already have; kept optional so a caller that
    # doesn't need it pays nothing.
    frontier_cells = find_frontier_cells(belief_map)
    if not frontier_cells:
        return np.zeros(FRONTIER_SUMMARY_D, dtype=np.float32)

    clusters = cluster_frontiers(frontier_cells)
    row, col = position
    # Nearest frontier cell by Manhattan distance; ties broken by (r, c) so the
    # chosen bearing is deterministic.
    nearest = min(frontier_cells, key=lambda cell: (abs(cell[0] - row) + abs(cell[1] - col), cell))
    nr, nc = nearest
    nearest_dist = (abs(nr - row) + abs(nc - col)) / float(height + width)

    return np.array(
        [
            min(len(frontier_cells) / float(height * width), 1.0),  # 0 frontier-cell density
            min(len(clusters) / float(MAX_FRONTIERS), 1.0),         # 1 cluster count
            min(nearest_dist, 1.0),                                 # 2 distance to nearest
            (nr - row) / float(height),                             # 3 bearing (row) in [-1, 1]
            (nc - col) / float(width),                              # 4 bearing (col) in [-1, 1]
            1.0,                                                    # 5 any-frontier flag
        ],
        dtype=np.float32,
    )


def build_comms_observation(
    *,
    belief_map: np.ndarray,
    sensed_mask: np.ndarray,
    trust_map: np.ndarray,
    position: Position,
    delivered_messages: Iterable[DeliveredMessage],
    comms_stats: CommsStats,
    tick: int,
    include_frontiers: bool = True,
    tick_scale: float = DEFAULT_TICK_SCALE,
    byte_scale: float = DEFAULT_BYTE_SCALE,
) -> CommsObservation:
    # Build one robot's fixed-shape comms observation. Pure and deterministic:
    # given the same inputs (in any order) it returns byte-identical tensors.
    height, width = belief_map.shape

    # -- robot_state (ROBOT_D,) ------------------------------------------------
    known_frac = float(np.count_nonzero(belief_map != UNKNOWN)) / float(height * width)
    sensed_frac = float(np.count_nonzero(sensed_mask)) / float(height * width)
    robot_state = np.array(
        [
            position[0] / float(max(height - 1, 1)),  # 0 normalized row
            position[1] / float(max(width - 1, 1)),   # 1 normalized col
            known_frac,                               # 2 fraction of map known (any source)
            sensed_frac,                              # 3 fraction sensed first-hand
        ],
        dtype=np.float32,
    )

    # -- message_features (MAX_MESSAGES, MESSAGE_D) + mask ---------------------
    ordered = sorted(delivered_messages, key=_message_sort_key)
    # Overflow keeps the FRESHEST messages (tail of the ascending sort) but leaves
    # them in deterministic order; padding rows stay all-zeros with mask 0.
    if len(ordered) > MAX_MESSAGES:
        ordered = ordered[-MAX_MESSAGES:]

    message_features = np.zeros((MAX_MESSAGES, MESSAGE_D), dtype=np.float32)
    message_mask = np.zeros(MAX_MESSAGES, dtype=np.float32)
    for i, m in enumerate(ordered):
        message_features[i] = _message_row(m, belief_map, position, height, width)
        message_mask[i] = 1.0

    # -- resource_features (RESOURCE_D,) --------------------------------------
    resource_features = np.array(
        [
            min(tick / tick_scale, 1.0),                                    # 0 elapsed-time budget
            min(comms_stats.messages_transmitted / float(max(tick, 1)), 1.0),  # 1 send rate/tick
            min(comms_stats.payload_bytes_transmitted / byte_scale, 1.0),  # 2 payload bytes spent
            float(trust_map.mean()),                                       # 3 mean comms trust
        ],
        dtype=np.float32,
    )

    # -- frontier_summary (FRONTIER_SUMMARY_D,) -------------------------------
    if include_frontiers:
        frontier_summary = _frontier_summary(belief_map, position, height, width)
    else:
        frontier_summary = np.zeros(FRONTIER_SUMMARY_D, dtype=np.float32)

    # Freeze every tensor: an observation is read-only at the policy edge.
    for arr in (robot_state, message_features, message_mask, resource_features, frontier_summary):
        arr.flags.writeable = False

    return CommsObservation(
        robot_state=robot_state,
        message_features=message_features,
        message_mask=message_mask,
        resource_features=resource_features,
        frontier_summary=frontier_summary,
    )
