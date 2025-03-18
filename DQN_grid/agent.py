import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x) 

class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=1e-3,
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.998,
                 memory_size=10000, batch_size=64, target_update=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.memory = deque(maxlen=memory_size) #经验缓冲区的大小
        self.learn_step_counter = 0
        
        # 创建在线网络和目标网络
        self.q_network = DQN(state_dim, action_dim, hidden_dim) #Q网络用于预测某个状态对应全部动作的Q值
        self.target_network = DQN(state_dim, action_dim, hidden_dim) #目标网络利用下一个状态预测当前的目标Q值（理想Q）
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
    def choose_action(self, state):
        """选择动作，使用epsilon-greedy策略"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))
    
    def update_epsilon(self):
        """更新探索率"""
        # - self.epsilon ：当前的探索率（初始值为1.0）
        # - self.epsilon_min ：探索率的最小值（0.01）
        # - self.epsilon_decay ：探索率的衰减系数，衰减系数越大下降越慢
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def learn(self):
        """从经验回放中学习"""
        if len(self.memory) < self.batch_size:
            return
        
        # 从经验回放中随机采样
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # action早在收集数据集到经验缓冲区的时候就已经选好了
        # self.q_network(states)更新了当前状态下四个方向的Q值，但是实际上只有一个方向是我们真实运动了的
        # 所以要gather(1, actions.unsqueeze(1))，只保留了我们真实action的那个方向的索引，依次来选取Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        # 关闭梯度计算，因为目标Q值不需要梯度
        with torch.no_grad():
        # 1. self.target_network(next_states) 返回的张量形状是 [batch_size, action_dim] ，例如：
        # tensor([[0.1, 0.3, 0.2, 0.4],    # 第一个状态的4个动作的Q值
        #         [0.2, 0.5, 0.1, 0.3],    # 第二个状态的4个动作的Q值
        #         [0.4, 0.1, 0.3, 0.2]])   # 第三个状态的4个动作的Q值
        # max(1)表示在行方向上取最大值，即取最大动作值
        # [0]表示只取最大值不返回索引
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失并更新Q网络，但不更新Q_target网络
        # 清空之前的梯度，计算损失，反向传播，更新网络参数
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 定期更新Q_target网络，使得Q_target网络和Q网络参数相同
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0: #每隔10次学习，将Q网络的参数拷贝至目标网络
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()