import numpy as np
import matplotlib.pyplot as plt
import torch

class GridWorld:
    def __init__(self, size=5, obstacles=None):
        self.size = size
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)
        self.n_features = 2  # x, y coordinates
        
        # 创建网格世界
        self.grid = np.zeros((size, size))
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        
        # 设置障碍物
        if obstacles is None:
            self.obstacles = [(1, 1), (2, 2), (3, 1), (1, 3), (3, 3), (0, 2)]
        else:
            self.obstacles = obstacles
            
        for obs in self.obstacles:
            self.grid[obs] = 1 #有障碍物的地方标记为1，其他地方为0
            
        self.agent_pos = self.start
        self.trajectory = [self.start]  # 添加轨迹列表
        self.best_trajectory = []  # 添加最佳轨迹记录

    def step(self, action):
        """执行一步动作"""
        x, y = self.agent_pos
        
        # 根据动作更新位置
        if action == 0:   # up
            x = max(0, x-1)
        elif action == 1: # down
            x = min(self.size-1, x+1)
        elif action == 2: # left
            y = max(0, y-1)
        elif action == 3: # right
            y = min(self.size-1, y+1)
            
        new_pos = (x, y)
        
        # 计算奖励
        # 调整奖励函数的设置，使得碰障碍物和碰到边界会获得较大的负面奖励
        # 给予到达终点更大的奖励值，鼓励到达终点
        if new_pos == self.goal:
            reward = 10000
            done = True
        elif new_pos in self.obstacles:
            reward = -50
            done = True
            new_pos = self.agent_pos  # 碰到障碍物回到起点
        elif new_pos[0] < 0 or new_pos[0] >= self.size or new_pos[1] < 0 or new_pos[1] >= self.size:
            reward = -50
            done = True
            new_pos = self.agent_pos  # 碰到边界回到起点
        else:
            reward = -1
            done = False
            
        self.agent_pos = new_pos
        self.trajectory.append(new_pos)  # 记录新位置
        return np.array(new_pos), reward, done

    def reset(self):
        """重置环境"""
        self.agent_pos = self.start
        self.trajectory = [self.start]  # 重置轨迹
        return np.array(self.agent_pos)

    def render(self, ax=None, show_arrows=True, agent=None):
        """可视化当前状态"""
        # 如果当前没有画布那么创建新的画布，如果有那么清楚现有内容
        if ax is None:
            plt.cla()
            ax = plt.gca()
        else:
            ax.cla()
            
        # 设置画布样式、大小
        ax.set_xlim(-0.5, self.size-0.5)
        ax.set_ylim(-0.5, self.size-0.5)
        ax.set_facecolor('#F8F9FA')  # 设置背景色
        
        # 绘制网格，使用更专业的样式
        ax.grid(True, linestyle='--', color='#E9ECEF', alpha=0.7)
        ax.set_xticks(np.arange(-0.5, self.size, 1))
        ax.set_yticks(np.arange(-0.5, self.size, 1))
        
        # 设置边框样式
        for spine in ax.spines.values():
            spine.set_color('#CED4DA')
            spine.set_linewidth(1.5)
        
        # 绘制障碍物，使用专业的配色
        for obs in self.obstacles:
            rect = plt.Rectangle((obs[1]-0.5, obs[0]-0.5), 1, 1,
                              facecolor='#ADB5BD',
                              edgecolor='#6C757D',
                              alpha=0.8)
            ax.add_patch(rect)
        
        # 绘制起点和终点，使用醒目的样式
        start = plt.Circle((self.start[1], self.start[0]), 0.3,
                          color='#2ECC71',
                          alpha=0.8,
                          label='Start')
        goal = plt.Circle((self.goal[1], self.goal[0]), 0.3,
                         color='#E74C3C',
                         alpha=0.8,
                         label='Goal')
        ax.add_patch(start)
        ax.add_patch(goal)
        
        # 绘制轨迹和箭头
        if len(self.trajectory) > 1:
            trajectory = np.array(self.trajectory)
            # 绘制路径线
            ax.plot(trajectory[:, 1], trajectory[:, 0],
                    color='#3498DB',
                    linestyle='--',
                    linewidth=2,
                    alpha=0.7,
                    zorder=1)
            
            # 添加箭头
            if show_arrows:
                for i in range(len(trajectory)-1):
                    dx = trajectory[i+1][1] - trajectory[i][1]
                    dy = trajectory[i+1][0] - trajectory[i][0]
                    ax.arrow(trajectory[i][1], trajectory[i][0], dx, dy,
                            head_width=0.2,
                            head_length=0.2,
                            fc='#3498DB',
                            ec='#3498DB',
                            alpha=0.8,
                            linewidth=2,
                            length_includes_head=True,
                            zorder=2)
        
        # 绘制Q值箭头
        if agent is not None:
            arrow_scale = 0.3  # 调整箭头大小
            for i in range(self.size):
                for j in range(self.size):
                    if (i, j) not in self.obstacles:
                        state = np.array([i, j])
                        with torch.no_grad():
                            q_values = agent.q_network(torch.FloatTensor(state).unsqueeze(0)).squeeze().numpy()
                        
                        # 修改归一化方式，确保最小值不会太小
                        q_min, q_max = q_values.min(), q_values.max()
                        if q_max > q_min:
                            q_values = (q_values - q_min) / (q_max - q_min)
                            # 将值范围从[0,1]调整到[0.2,1]
                            q_values = q_values * 0.8 + 0.2
                        
                        # 使用更细致的箭头样式
                        arrow_style = dict(
                            head_width=0.15,
                            head_length=0.15,
                            fc='#FF6B6B',
                            ec='#FF6B6B',
                            alpha=0.6,
                            linewidth=1.5,
                            length_includes_head=True
                        )
                        
                        # 绘制四个方向的箭头
                        ax.arrow(j, i, 0, -q_values[0] * arrow_scale, **arrow_style)  # 上
                        ax.arrow(j, i, 0, q_values[1] * arrow_scale, **arrow_style)   # 下
                        ax.arrow(j, i, -q_values[2] * arrow_scale, 0, **arrow_style)  # 左
                        ax.arrow(j, i, q_values[3] * arrow_scale, 0, **arrow_style)   # 右
        
        # 绘制智能体当前位置
        agent_marker = plt.Circle((self.agent_pos[1], self.agent_pos[0]), 0.25,
                                color='#3498DB',
                                alpha=0.9,
                                label='Agent')
        ax.add_patch(agent_marker)
        
        # 设置标题和图例
        ax.set_title('Grid World Environment', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)
        
        # 设置轴标签
        ax.set_xlabel('X Position', fontsize=10)
        ax.set_ylabel('Y Position', fontsize=10)
        
        # 调整布局
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1) #让该窗口停止0.1秒，因为程序一直在运行render函数一直被调用，人的视觉上就是连贯的了

    def save_best_trajectory(self):
        """只在智能体找到一条更好的路径（即获得更高奖励）且该路径是连续的、没有回头或重复的情况下才保存为最佳轨迹。"""
        # 检查轨迹的连续性
        is_continuous = True
        for i in range(len(self.trajectory)-1):
            curr_pos = self.trajectory[i]
            next_pos = self.trajectory[i+1]
            # 检查相邻位置是否连续（曼哈顿距离为1）
            if abs(curr_pos[0] - next_pos[0]) + abs(curr_pos[1] - next_pos[1]) != 1:
                is_continuous = False
                break
        
        # 检查是否有重复访问的位置
        unique_positions = set(self.trajectory)
        if is_continuous and len(unique_positions) == len(self.trajectory):
            self.best_trajectory = self.trajectory.copy()

    def render_best_trajectory(self, ax=None):
        """渲染最佳轨迹"""
        if not self.best_trajectory:
            return
            
        if ax is None:
            plt.figure(figsize=(8, 8), dpi=100)
            ax = plt.gca()
            
        ax.clear()
        ax.set_xlim(-0.5, self.size-0.5)
        ax.set_ylim(-0.5, self.size-0.5)
        
        # 设置网格样式
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xticks(np.arange(-0.5, self.size, 1))
        ax.set_yticks(np.arange(-0.5, self.size, 1))
        
        # 绘制障碍物，使用更专业的样式
        for obs in self.obstacles:
            rect = plt.Rectangle((obs[1]-0.5, obs[0]-0.5), 1, 1,
                              facecolor='#A0A0A0',
                              edgecolor='#808080',
                              alpha=0.8)
            ax.add_patch(rect)
        
        # 绘制起点和终点，使用更醒目的样式
        start = plt.Circle((self.start[1], self.start[0]), 0.3,
                          color='#2ECC71',
                          alpha=0.8,
                          label='Start')
        goal = plt.Circle((self.goal[1], self.goal[0]), 0.3,
                         color='#E74C3C',
                         alpha=0.8,
                         label='Goal')
        ax.add_patch(start)
        ax.add_patch(goal)
        
        # 绘制最佳轨迹
        if self.best_trajectory:
            trajectory = np.array(self.best_trajectory)
            
            # 绘制路径线
            ax.plot(trajectory[:, 1], trajectory[:, 0],
                    color='#3498DB',
                    linestyle='-',
                    linewidth=2.5,
                    alpha=0.8,
                    zorder=1)
            
            # 添加箭头
            for i in range(len(trajectory)-1):
                dx = trajectory[i+1][1] - trajectory[i][1]
                dy = trajectory[i+1][0] - trajectory[i][0]
                ax.arrow(trajectory[i][1], trajectory[i][0], dx, dy,
                        head_width=0.2,
                        head_length=0.2,
                        fc='#3498DB',
                        ec='#3498DB',
                        alpha=0.8,
                        linewidth=2.5,
                        length_includes_head=True,
                        zorder=2)
        
        # 设置标题和图例
        ax.set_title('Optimal Path Trajectory', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)
        
        # 设置轴标签
        ax.set_xlabel('X Position', fontsize=10)
        ax.set_ylabel('Y Position', fontsize=10)
        
        # 调整布局
        plt.tight_layout()
        ax.legend()
        ax.set_title('Best Trajectory', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.legend()