"""
dqn.py
Double Dueling DQN with NoisyNet网络架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class NoisyLinear(nn.Module):
    """Noisy线性层实现"""
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """
        Noisy线性层初始化
        
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            std_init: 噪声标准差初始化值
        """
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # 权重参数
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        
        # 偏置参数
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        # 初始化参数
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """重置参数初始值"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """重置噪声"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # 外积产生噪声
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """生成噪声向量"""
        noise = torch.randn(size)
        return noise.sign().mul(noise.abs().sqrt())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.training:
            # 训练时添加噪声
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return F.linear(x, weight, bias)
        else:
            # 评估时不添加噪声
            return F.linear(x, self.weight_mu, self.bias_mu)

class DuelingDQN(nn.Module):
    """Dueling DQN网络架构"""
    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int, config: DQNConfig):
        """
        Dueling DQN初始化
        
        Args:
            input_shape: 输入形状 (通道, 高度, 宽度)
            n_actions: 动作空间大小
            config: 配置参数
        """
        super(DuelingDQN, self).__init__()
        self.input_channels = input_shape[0]
        self.n_actions = n_actions
        
        # 卷积层提取空间特征
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 计算卷积输出尺寸
        conv_out_size = self._get_conv_out(input_shape)
        
        # 使用Noisy网络或标准全连接层
        LinearLayer = NoisyLinear if config.NOISY_NET else nn.Linear
        
        # Dueling架构: 价值流和优势流
        self.value_stream = nn.Sequential(
            LinearLayer(conv_out_size, 512),
            nn.ReLU(),
            LinearLayer(512, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            LinearLayer(conv_out_size, 512),
            nn.ReLU(),
            LinearLayer(512, n_actions)
        )
    
    def _get_conv_out(self, shape: Tuple[int, int, int]) -> int:
        """计算卷积层输出尺寸"""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 卷积特征提取
        conv_out = self.conv(x).view(x.size(0), -1)
        
        # Dueling网络流
        value = self.value_stream(conv_out)
        advantage = self.advantage_stream(conv_out)
        
        # 组合价值流和优势流
        return value + advantage - advantage.mean(dim=1, keepdim=True)
    
    def reset_noise(self):
        """重置所有噪声层"""
        for layer in self.children():
            if hasattr(layer, 'reset_noise'):
                layer.reset_noise()
    
    def act(self, state: torch.Tensor) -> int:
        """选择动作 (贪婪策略)"""
        with torch.no_grad():
            q_values = self(state.unsqueeze(0))
            return q_values.max(1)[1].item()

class DoubleDuelingDQN:
    """Double Dueling DQN封装"""
    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int, config: DQNConfig):
        # 创建当前网络和目标网络
        self.policy_net = DuelingDQN(input_shape, n_actions, config)
        self.target_net = DuelingDQN(input_shape, n_actions, config)
        
        # 初始化目标网络
        self.update_target()
        self.target_net.eval()  # 目标网络不更新
    
    def update_target(self):
        """更新目标网络参数"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def reset_noise(self):
        """重置噪声层"""
        self.policy_net.reset_noise()