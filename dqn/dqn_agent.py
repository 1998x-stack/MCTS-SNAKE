"""
dqn_agent.py
Double Dueling DQN智能体实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple
from loguru import logger
from tensorboardX import SummaryWriter
from .dqn import DoubleDuelingDQN
from .config import DQNConfig

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """添加经验到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """从缓冲区采样批次数据"""
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        """返回当前缓冲区大小"""
        return len(self.buffer)

class DQNAgent:
    """DQN智能体"""
    def __init__(self, state_shape: Tuple[int, int, int], n_actions: int, config: DQNConfig):
        """
        智能体初始化
        
        Args:
            state_shape: 状态形状 (通道, 高度, 宽度)
            n_actions: 动作空间大小
            config: 配置参数
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.config = config
        
        # 创建DQN网络
        self.model = DoubleDuelingDQN(state_shape, n_actions, config)
        self.policy_net = self.model.policy_net.to(self.device)
        self.target_net = self.model.target_net.to(self.device)
        
        # 创建优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LR)
        
        # 创建经验回放缓冲区
        self.memory = ReplayBuffer(config.REPLAY_SIZE)
        
        # 探索率设置
        self.epsilon = config.START_EPSILON
        self.epsilon_decay = (config.START_EPSILON - config.END_EPSILON) / config.EPSILON_DECAY
        self.steps_done = 0
        
        # 日志记录
        self.writer = SummaryWriter()
        self.losses = []
        self.rewards = []
        
        logger.info(f"智能体初始化完成 | 设备: {self.device}")
    
    def select_action(self, state: np.ndarray) -> int:
        """选择动作 (ε-贪婪策略)"""
        self.steps_done += 1
        
        # 使用NoisyNet时不需要ε贪婪
        if self.config.NOISY_NET:
            # 确保在训练模式以启用噪声
            self.policy_net.train()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.policy_net.act(state_tensor)
        
        # 标准ε贪婪策略
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                self.policy_net.eval()
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                return self.policy_net.act(state_tensor)
    
    def update_epsilon(self):
        """更新探索率"""
        if not self.config.NOISY_NET and self.epsilon > self.config.END_EPSILON:
            self.epsilon -= self.epsilon_decay
            self.writer.add_scalar('Epsilon', self.epsilon, self.steps_done)
    
    def push_memory(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """添加经验到记忆"""
        self.memory.push(state, action, reward, next_state, done)
    
    def optimize_model(self):
        """优化模型参数"""
        if len(self.memory) < self.config.INITIAL_MEMORY:
            return 0.0
        
        # 从回放缓冲区采样
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.BATCH_SIZE)
        
        # 转换为PyTorch张量
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # 计算当前Q值
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值 (Double DQN)
        with torch.no_grad():
            # 使用策略网络选择最佳动作
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            # 使用目标网络评估Q值
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1 - dones) * self.config.GAMMA * next_q
        
        # 计算Huber损失
        loss = F.smooth_l1_loss(current_q, target_q)
        self.losses.append(loss.item())
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # 重置噪声层
        if self.config.NOISY_NET:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()
        
        return loss.item()
    
    def update_target_net(self):
        """更新目标网络"""
        if self.steps_done % self.config.UPDATE_FREQ == 0:
            self.model.update_target()
            logger.debug(f"目标网络更新 | 步数: {self.steps_done}")
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save(self.policy_net.state_dict(), path)
        logger.info(f"模型已保存: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        self.policy_net.load_state_dict(torch.load(path))
        self.update_target()
        logger.info(f"模型已加载: {path}")
    
    def log_step(self, reward: float):
        """记录当前步"""
        self.rewards.append(reward)
        if self.steps_done % self.config.LOG_FREQ == 0:
            avg_reward = np.mean(self.rewards[-self.config.LOG_FREQ:])
            avg_loss = np.mean(self.losses[-self.config.LOG_FREQ:]) if self.losses else 0.0
            
            self.writer.add_scalar('Training/Reward', avg_reward, self.steps_done)
            self.writer.add_scalar('Training/Loss', avg_loss, self.steps_done)
            self.writer.add_scalar('Training/Epsilon', self.epsilon, self.steps_done)
            
            logger.info(f"步数: {self.steps_done} | 平均奖励: {avg_reward:.2f} | 平均损失: {avg_loss:.4f} | ε: {self.epsilon:.3f}")
    
    def close(self):
        """关闭智能体"""
        self.writer.close()
        logger.info("智能体关闭")