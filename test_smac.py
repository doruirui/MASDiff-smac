import numpy as np
import random
from smac.env import StarCraft2Env

def test_smac():
    env = StarCraft2Env(map_name="3m", step_mul=8, difficulty="7", seed=42)
    env.reset()
    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    episode_limit = env_info["episode_limit"]

    for step in range(episode_limit):
        # 获取每个智能体的合法动作掩码
        avail_actions_mask = [env.get_avail_agent_actions(i) for i in range(n_agents)]
        # 打印掩码（可选）
        # print(f"Step {step}: masks = {avail_actions_mask}")

        # 从掩码中提取合法动作编号
        actions = []
        for i, mask in enumerate(avail_actions_mask):
            legal = [act_id for act_id, val in enumerate(mask) if val == 1]
            if len(legal) == 0:
                print(f"Warning: Agent {i} has no legal actions at step {step}")
                action = 0
            else:
                action = random.choice(legal)   # 使用 Python random 或 np.random.choice
            actions.append(action)

        print(f"Step {step}: actions = {actions}")
        reward, done, _ = env.step(actions)
        if done:
            print(f"Episode finished at step {step}, total reward: {reward}")
            break
        if step > 500:
            break
    env.close()

if __name__ == "__main__":
    test_smac()