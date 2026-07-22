from src.policy.move_random_policy import move_random
from src.policy.move_toward_unknown_policy import move_toward_unknown

POLICIES = {
    "move_random": move_random,
    "move_toward_unknown": move_toward_unknown,
}
