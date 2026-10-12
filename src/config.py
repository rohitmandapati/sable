# Fixed-shape bounds for the ActorObservation/ActorAction contracts
# Static shapes are a hard requirement and these are the single
# source of truth so the space, the feature builder, and the policy always agree.

MAX_FRONTIERS = 16
MAX_NEIGHBORS = 8
MAX_MESSAGES = 16

ROBOT_D = 4     # robot_state feature width
FRONTIER_D = 6  # per-frontier feature width
NEIGHBOR_D = 4  # reserved; unused until neighbor sensing
MESSAGE_D = 8   # per-delivered-message feature width (comms_observation)
META_D = 4      # reserved; message metadata, unused until comms

RESOURCE_D = 4          # budget/resource (comms-accounting) feature width
FRONTIER_SUMMARY_D = 6  # compact global frontier-summary feature width

HIDDEN_D = 64   # policy-managed GRU state; NOT part of the env observation