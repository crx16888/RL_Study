import numpy as np
import matplotlib.pyplot as plt
import torch
from env import GridWorld
from agent import DQNAgent

# 训练参数
# MAX_STEPS要让智能体足够找到最优路径，并且避免设置过大智能体过度游走浪费计算资源；EPISODES的设置要让智能体有足够的机会去探索学习并且Q值收敛
EPISODES = 2000
MAX_STEPS = 50
GRID_SIZE = 5
# - EPISODES = 500 ：
# - 增加训练回合数可以让智能体有更多机会探索和学习
# - 由于环境相对简单（5x5网格），500回合足够让Q值收敛
# - 可以更好地观察奖励曲线的趋势和学习效果

# - MAX_STEPS = 50 ：
# - 在5x5的网格中，从起点到终点的最优路径不会超过10步
# - 设置50步足够智能体在一个回合内探索和找到目标
# - 过大的步数（如100）可能导致：
#   - 智能体在一个回合中过度游走
#   - 浪费计算资源
#   - 增加学习无效经验的可能性

# 创建环境和智能体
env = GridWorld(size=GRID_SIZE)
agent = DQNAgent(state_dim=env.n_features,
                action_dim=env.n_actions,
                hidden_dim=64,
                lr=1e-4,
                gamma=0.99,
                epsilon=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.998)

# 记录训练数据
episode_rewards = []
total_losses = []

# 创建图形窗口
plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))
plt.subplots_adjust(wspace=0.3, hspace=0.3) #设置子图间距

def update_visualization(ax1, ax2, ax3, episode, episode_rewards, total_losses, env, agent):
    # 只在关键回合（前50回合、每50回合和最后50回合）显示可视化
    if episode < 50 or episode % 50 == 0 or episode >= EPISODES - 50:
        env.render(ax1, show_arrows=True, agent=agent)
        ax1.set_title(f'Episode {episode+1}')
        
        # 绘制奖励曲线
        ax2.clear()
        ax2.plot(episode_rewards)
        ax2.set_title('Episode Rewards')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax2.grid(True)
        
        # 绘制损失曲线
        if total_losses:
            ax3.clear()
            ax3.plot(total_losses)
            ax3.set_title('Training Loss')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Loss')
            ax3.grid(True)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

# 训练循环
best_reward = float('-inf')
for episode in range(EPISODES):
    state = env.reset()
    episode_reward = 0
    episode_loss = []
    
    for step in range(MAX_STEPS):
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        
        loss = agent.learn()
        if loss is not None:
            episode_loss.append(loss)
        
        state = next_state
        episode_reward += reward
        
        if done:
            if reward > 0 and episode_reward > best_reward:
                best_reward = episode_reward
                env.save_best_trajectory()
            break
    
    agent.update_epsilon()
    episode_rewards.append(episode_reward)
    if episode_loss:
        total_losses.append(np.mean(episode_loss))
    
    update_visualization(ax1, ax2, ax3, episode, episode_rewards, total_losses, env, agent)
    print(f'Episode: {episode+1}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}')

# 保存训练结果
plt.savefig('C:/Users/95718/Desktop/vscode/RL/DQN_grid/training_results.png')
plt.ioff()
plt.show()

# 训练结束后显示最佳路径
plt.figure(figsize=(10, 10))
env.render_best_trajectory()
plt.savefig('C:/Users/95718/Desktop/vscode/RL/DQN_grid/best_trajectory.png')
plt.show()
