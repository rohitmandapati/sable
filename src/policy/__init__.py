from policy.move_random_policy import move_random
from policy.move_toward_unknown_policy import move_toward_unknown, move_toward_unknown_bfs

POLICIES = {
    "move_random": move_random,
    "move_toward_unknown": move_toward_unknown,
    "move_toward_unknown_bfs": move_toward_unknown_bfs,
}
