"""Quick test: sigils vs no-sigils in combat sim."""
from .vec_sim import VecPvPSim
from .config import NUM_ACTIONS
import numpy as np

s = VecPvPSim(256, seed=42)
s.reset_all()
kills_a, kills_b = 0, 0

for _ in range(2000):
    a = np.zeros((256, NUM_ACTIONS), dtype=np.int8)
    a[:, 0] = 1   # forward
    a[:, 26] = 1  # engage
    b = a.copy()
    # B uses all sigils every tick (cooldown gates actual activation)
    b[:, 14] = 1  # brace
    b[:, 15] = 1  # cleopatra
    b[:, 16] = 1  # quick sand
    b[:, 17] = 1  # nile's grace

    obs, rew, done, info = s.step(a, b)
    kills_a += (info['b_health'] <= 0).sum()
    kills_b += (info['a_health'] <= 0).sum()

print(f"A kills (no sigils): {kills_a}")
print(f"B kills (with sigils): {kills_b}")
print(f"Sigil advantage: {kills_b / max(kills_a, 1):.1f}x")
