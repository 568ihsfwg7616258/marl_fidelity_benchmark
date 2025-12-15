# src/env
from pettingzoo.mpe import simple_spread_v3
import numpy as np

def create_lbf_env(n_agents=3):
    """Simple and well-known environment Level-Based Foraging for MARL"""
    env = simple_spread_v3.env(
        N=n_agents,
        local_ratio=0.5,
        max_cycles=100,
        continuous_actions=False
    )
    env.reset()
    return env

if __name__ == "__main__":
    env = create_lbf_env()
    print("LBF environment successfully created!")
    print("Number of agents:", env.num_agents)
    print("Observation of agent 0:", len(env.observe(env.agents[0])))