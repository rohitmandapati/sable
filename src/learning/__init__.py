# SABLE learning layer.
#
# The MAPPO-facing side of the sim: feature builders and (later) the learned
# comms policy + PPO stack. Today it holds only the comms observation builder --
# a pure, deterministic converter from robot state to fixed-shape numpy tensors.
# No PyTorch/PPO yet; nothing here touches movement or mutates world state.

from learning.comms_observation import CommsObservation, build_comms_observation

__all__ = [
    "CommsObservation",
    "build_comms_observation",
]
