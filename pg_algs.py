import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple

np.random.seed(42)
torch.manual_seed(42)

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class BaselineNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def compute_total_return(rewards, gamma=0.99):
    G = sum(gamma**t * r for t, r in enumerate(rewards))
    return [G] * len(rewards)

def compute_reward_to_go(rewards, gamma=0.99):
    rtg = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        rtg.insert(0, R)
    return rtg

def collect_trajectories(env, policy, batch_size):
    Trajectory = namedtuple("Trajectory", ["states", "actions", "rewards"])
    trajectories = []
    for _ in range(batch_size):
        state = env.reset()[0]
        states, actions, rewards = [], [], []
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state)
            probs = policy(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            states.append(state.tolist())
            actions.append(action.item())
            state, reward, done, _, _ = env.step(action.item())
            rewards.append(reward)

        trajectories.append(Trajectory(states, actions, rewards))
    return trajectories

def compute_policy_loss(policy, trajectories, returns_list):
    loss = []
    for traj, returns in zip(trajectories, returns_list):
        for state, action, G in zip(traj.states, traj.actions, returns):
            state_tensor = torch.FloatTensor(state)
            probs = policy(state_tensor)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(torch.tensor(action))
            loss.append(-log_prob * G)
    return torch.stack(loss).sum()

def evaluate_policy(policy, env, episodes=100):
    rewards = []
    success_count = 0
    for _ in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                probs = policy(state_tensor)
            action = torch.argmax(probs).item()
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        if total_reward >= 195:
            success_count += 1
    return np.mean(rewards), np.std(rewards), success_count

# --- Training ---
env = gym.make("CartPole-v1")
episodes = 200
batch_size = 5
gamma = 0.99

policy_vanilla = PolicyNetwork()
policy_rtg = PolicyNetwork()
policy_baseline = PolicyNetwork()
baseline_net = BaselineNetwork()

opt_vanilla = optim.Adam(policy_vanilla.parameters(), lr=1e-2)
opt_rtg = optim.Adam(policy_rtg.parameters(), lr=1e-2)
opt_baseline = optim.Adam(list(policy_baseline.parameters()) + list(baseline_net.parameters()), lr=1e-2)

vanilla_rewards, rtg_rewards, baseline_rewards = [], [], []

for ep in range(episodes):
    trajectories = collect_trajectories(env, policy_vanilla, batch_size)

    v_returns = [compute_total_return(t.rewards, gamma) for t in trajectories]
    loss_v = compute_policy_loss(policy_vanilla, trajectories, v_returns)
    opt_vanilla.zero_grad(); loss_v.backward(); opt_vanilla.step()

    rtg_returns = [compute_reward_to_go(t.rewards, gamma) for t in trajectories]
    loss_rtg = compute_policy_loss(policy_rtg, trajectories, rtg_returns)
    opt_rtg.zero_grad(); loss_rtg.backward(); opt_rtg.step()

    all_states, all_returns = [], []
    baseline_returns = []
    for traj in trajectories:
        rtg = compute_reward_to_go(traj.rewards, gamma)
        all_states.extend(traj.states)
        all_returns.extend(rtg)
        baseline_returns.append(rtg)

    state_tensor = torch.FloatTensor(all_states)
    returns_tensor = torch.FloatTensor(all_returns)
    pred_values = baseline_net(state_tensor)
    advantages_tensor = returns_tensor - pred_values.detach()

    loss_b_value = nn.functional.mse_loss(pred_values, returns_tensor)

    start = 0
    grouped_advantages = []
    for traj in trajectories:
        n = len(traj.rewards)
        grouped_advantages.append(advantages_tensor[start:start+n])
        start += n

    loss_b_policy = compute_policy_loss(policy_baseline, trajectories, grouped_advantages)
    loss_b_total = loss_b_policy + 0.5 * loss_b_value
    opt_baseline.zero_grad(); loss_b_total.backward(); opt_baseline.step()

    avg_r = lambda ts: np.mean([sum(t.rewards) for t in ts])
    vanilla_rewards.append(avg_r(trajectories))
    rtg_rewards.append(avg_r(trajectories))
    baseline_rewards.append(avg_r(trajectories))

    if (ep+1) % 10 == 0:
        print(f"Episode {ep+1}: Vanilla={vanilla_rewards[-1]:.1f}, RTG={rtg_rewards[-1]:.1f}, Baseline={baseline_rewards[-1]:.1f}")

# Save models
torch.save(policy_vanilla.state_dict(), "vanilla.pt")
torch.save(policy_rtg.state_dict(), "rtg.pt")
torch.save(policy_baseline.state_dict(), "baseline.pt")

# Evaluate
print("\n🔍 Evaluating Trained Policies on 100 Test Episodes...")
eval_env = gym.make("CartPole-v1")
results = {}
for name, policy in zip(["Vanilla", "Reward-to-Go", "Baseline"], [policy_vanilla, policy_rtg, policy_baseline]):
    mean_r, std_r, success = evaluate_policy(policy, eval_env)
    results[name] = {"Mean Reward": mean_r, "Std Dev": std_r, "Success Count": success}
    print(f"\n{name} Policy:\n  Mean Reward: {mean_r:.2f}\n  Std Dev: {std_r:.2f}\n  Successes (>=195): {success}/100")

# Save logs
df_rewards = pd.DataFrame({
    "Episode": list(range(1, episodes + 1)),
    "Vanilla": vanilla_rewards,
    "Reward-to-Go": rtg_rewards,
    "Baseline": baseline_rewards
})
df_rewards.to_excel("training_rewards_log.xlsx", index=False)

pd.DataFrame(results).T.to_excel("evaluation_results.xlsx")
print("\n✅ Training and Evaluation Complete. Logs saved.")


