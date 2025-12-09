from pettingzoo.butterfly import pistonball_v6
# یا بهتر: Level-Based Foraging
from pettingzoo.mpe import simple_spread_v3

env = simple_spread_v3.env(N=3, local_ratio=0.5, max_cycles=100)
env.reset()
print("Observation space:", env.observation_spaces)
print("Action space:", env.action_spaces)