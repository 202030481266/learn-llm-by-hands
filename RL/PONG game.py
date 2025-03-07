# pip install gymnasium[atari] ale-py torch

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import pickle
import os

# 超参数
H = 200               # 隐藏层神经元数
batch_size = 128       # 可调整为2的幂次方
learning_rate = 1e-4
gamma = 0.99          # 奖励折扣因子
resume = True         # 是否从检查点恢复
test = False          # 测试模式关闭（Colab中不支持渲染）
save_file = 'pong_model_pytorch.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# 初始化模型
D = 80 * 80  # 输入维度
model = PolicyNetwork(D, H).to(device)

if resume and os.path.exists(save_file):
    model.load_state_dict(torch.load(save_file))
    print("Loaded model from checkpoint.")

optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-5)

# 预处理函数
def prepro(I):
    """ 将210x160x3图像预处理为80x80的一维 float 数组 """
    I = I[35:195]         # 裁剪
    I = I[::2, ::2, 0]     # 降采样：步长为2
    I[I == 144] = 0       # 消除背景1
    I[I == 109] = 0       # 消除背景2
    I[I != 0] = 1         # 其他部分置1
    return I.astype(np.float32).ravel()

# 折扣奖励函数
def discount_rewards(r):
    r = np.array(r)
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(r.size)):
        if r[t] != 0:
            running_add = 0  # 游戏边界，重置累加器（Pong特有）
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

# 创建环境（在Colab中不启用渲染）
env = gym.make("ale_py:ALE/Pong-v5")  # 不指定 render_mode

# 初始化变量
observation, _ = env.reset()
prev_x = None
# xs 用于记录观察（如果后续需要），rewards记录每步奖励，probs记录每步预测的概率，actions记录实际采取的动作标签（1代表action 2，0代表action 3）
xs, rewards, probs, actions = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

while True:
    # 预处理当前观察，并计算与上一帧的差值
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # 转为tensor并传入网络得到动作概率
    x_tensor = torch.FloatTensor(x).to(device)
    prob = model(x_tensor)

    # 根据概率选择动作，并记录对应的标签 y（1: 动作2, 0: 动作3）
    if test:
        if prob.item() > 0.5:
            action = 2
            y = 1
        else:
            action = 3
            y = 0
    else:
        if np.random.uniform() < prob.item():
            action = 2
            y = 1
        else:
            action = 3
            y = 0

    xs.append(x)
    probs.append(prob)
    actions.append(y)  # 记录当前采取的动作标签

    # 与环境交互
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    reward_sum += reward
    rewards.append(reward)

    if done:
        episode_number += 1

        # 计算折扣奖励并标准化
        discounted_rewards = discount_rewards(np.array(rewards))
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(device)

        if not test:
            optimizer.zero_grad()
            policy_loss = []
            # 注意：此处遍历每一步，使用记录下来的 prob、动作标签和对应折扣奖励
            for prob, y, r in zip(probs, actions, discounted_rewards):
                # 若 y==1 则取 log(prob)，否则取 log(1-prob)
                log_prob = torch.log(prob if y == 1 else 1 - prob)
                policy_loss.append(-log_prob * r)
            policy_loss = torch.stack(policy_loss).sum()
            policy_loss.backward()
            optimizer.step()

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print(f'ep {episode_number}: reward: {reward_sum}, running mean: {running_reward:.2f}')

        if episode_number % 100 == 0 and not test:
            torch.save(model.state_dict(), save_file)

        # 重置环境和存储变量
        reward_sum = 0
        observation, _ = env.reset()
        prev_x = None
        xs, rewards, probs, actions = [], [], [], []

    if reward != 0:
        print(f'ep {episode_number}: game finished, reward: {reward}' + (' !!!!!!!!' if reward == 1 else ''))