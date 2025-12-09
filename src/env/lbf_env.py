# src/env/lbf_env.py
from pettingzoo.mpe import simple_spread_v3
import numpy as np

def create_lbf_env(n_agents=3):
    """محیط ساده و معروف Level-Based Foraging برای MARL"""
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
    print("محیط LBF با موفقیت ساخته شد!")
    print("تعداد عامل‌ها:", env.num_agents)
    print("مشاهده عامل 0:", len(env.observe(env.agents[0])))