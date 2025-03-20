"""
Custom Gym Environment for Snake Game (实现)
- 支持高维状态观测 (RGB图像或特征向量)
- 包含完整异常处理和性能监控
"""

import gym
import numpy as np
from typing import Tuple, Dict, Any, Optional
from gym import spaces
import cv2
import time


class SnakeEnv(gym.Env):
    """贪吃蛇环境，符合Google代码规范与PEP 8/PEP 257标准"""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_size: int = 20, frame_rate: int = 30) -> None:
        """初始化环境参数
        
        Args:
            grid_size: 游戏网格尺寸 (默认20x20)
            frame_rate: 渲染帧率 (需求需支持动态调整)
        """
        super(SnakeEnv, self).__init__()
        
        # 环境参数验证
        if grid_size < 5:
            raise ValueError("网格尺寸必须≥5以保证游戏可行性")
        
        self.grid_size = grid_size
        self.frame_rate = frame_rate
        
        # 动作空间: 上下左右 (4个离散动作)
        self.action_space = spaces.Discrete(4)
        
        # 观测空间: 多通道特征 (需求)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(grid_size, grid_size, 3),
            dtype=np.uint8
        )
        
        # 初始化游戏状态
        self.snake_body = []  # 蛇身坐标列表
        self.snake_direction = 0  # 当前方向
        self.food_pos = None  # 食物位置
        self.total_reward = 0.0  # 累计奖励
        self.step_counter = 0  # 步数计数器
        
        # 性能监控
        self._last_render_time = 0.0

    def reset(self) -> np.ndarray:
        """重置环境至初始状态"""
        # 初始化蛇的位置 (居中)
        start_pos = (self.grid_size // 2, self.grid_size // 2)
        self.snake_body = [start_pos]
        
        # 随机生成食物
        self._generate_food()
        
        # 重置统计信息
        self.total_reward = 0.0
        self.step_counter = 0
        
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行动作并返回环境状态变化
        
        Args:
            action: 动作编码 (0-3对应上下左右)
            
        Returns:
            observation: 当前观测状态
            reward: 本步奖励值
            done: 是否终止
            info: 诊断信息
        """
        # 方向有效性检查
        self._validate_action(action)
        
        # 更新蛇头位置
        new_head = self._calculate_new_head(action)
        # 更新方向
        self.snake_direction = action  # 记录当前方向
        
        # 碰撞检测
        done = self._check_collision(new_head)
        reward = self._calculate_reward(new_head, done)
        
        # 更新蛇身状态
        if not done:
            self.snake_body.insert(0, new_head)
            if new_head == self.food_pos:
                self._generate_food()
            else:
                self.snake_body.pop()
        
        # 更新统计信息
        self.total_reward += reward
        self.step_counter += 1
        
        return self._get_observation(), reward, done, self._get_info()

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """渲染当前游戏状态"""
        # 帧率控制
        current_time = time.time()
        if current_time - self._last_render_time < 1/self.frame_rate:
            return
        self._last_render_time = current_time
        
        # 创建画布
        canvas = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        # 绘制蛇身
        for idx, (x, y) in enumerate(self.snake_body):
            color = (0, 255, 0) if idx == 0 else (0, 180, 0)  # 蛇头与蛇身颜色区分
            canvas[x, y] = color
            
        # 绘制食物
        canvas[self.food_pos] = (0, 0, 255)
        
        # 缩放显示
        scaled_canvas = cv2.resize(canvas, (400, 400), interpolation=cv2.INTER_NEAREST)
        
        if mode == 'human':
            cv2.imshow('Snake Game', scaled_canvas)
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return scaled_canvas
        return None

    def seed(self, seed: Optional[int] = None) -> None:
        """设置环境随机种子 (新增方法)"""
        seed_seq = np.random.SeedSequence(seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        # 食物生成器独立种子
        self._food_rng = np.random.default_rng(seed_seq.generate_state(1))
        return [seed]
        
    def close(self) -> None:
        """清理环境资源"""
        cv2.destroyAllWindows()

    # ------------------------- 内部工具方法 -------------------------
    def _generate_food(self) -> None:
        """生成新食物位置 (确保不与蛇身重叠)"""
        while True:
            new_food = (
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            )
            if new_food not in self.snake_body:
                self.food_pos = new_food
                break

    def _calculate_new_head(self, action: int) -> Tuple[int, int]:
        """根据动作计算新蛇头位置"""
        direction_map = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        dx, dy = direction_map[action]
        current_head = self.snake_body[0]
        return (current_head[0] + dx) % self.grid_size, (current_head[1] + dy) % self.grid_size

    def _check_collision(self, new_head: Tuple[int, int]) -> bool:
        """检测碰撞条件 (边界或自碰)"""
        # 检查是否碰到自身
        if new_head in self.snake_body[1:]:
            return True
        return False

    def _calculate_reward(self, new_head: Tuple[int, int], done: bool) -> float:
        """计算奖励函数 (奖励设计)"""
        if new_head == self.food_pos:
            return 5.0    # 吃到食物奖励
        if done:
            return -10.0  # 死亡惩罚
        # 鼓励靠近食物
        food_dist = abs(new_head[0]-self.food_pos[0]) + abs(new_head[1]-self.food_pos[1])
        return 1.0 / (food_dist + 1)  # 距离奖励

    def _get_observation(self) -> np.ndarray:
        """生成观测状态 (特征工程)"""
        # 创建三通道特征图
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        # 通道1: 蛇身位置 (255为蛇头, 128为蛇身)
        for idx, (x, y) in enumerate(self.snake_body):
            obs[x, y, 0] = 255 if idx == 0 else 128
            
        if self.food_pos is not None:  # 新增保护
            # 通道2: 食物位置 (255为食物)
            obs[self.food_pos[0], self.food_pos[1], 1] = 255
        
        # 通道3: 方向编码 (特征)
        obs[:, :, 2] = self.snake_direction * 64  # 确保使用最新方向值
        
        # 新增边界增强特征
        border_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        border_mask[0,:] = border_mask[-1,:] = border_mask[:,0] = border_mask[:,-1] = 1
        obs[:, :, 2] = np.where(border_mask, 128, obs[:, :, 2])
        return obs

    def _get_info(self) -> Dict[str, Any]:
        """获取诊断信息 (用于监控系统)"""
        return {
            'snake_length': len(self.snake_body),
            'total_reward': self.total_reward,
            'steps': self.step_counter,
            'food_position': self.food_pos
        }

    def _validate_action(self, action: int) -> None:
        """动作有效性检查 (边界条件处理)"""
        if action not in self.action_space:
            raise ValueError(f"无效动作: {action}. 有效动作范围: 0-3")
        
        # 防止180度转向
        opposite_map = {0:1, 1:0, 2:3, 3:2}  # 上<->下, 左<->右
        if opposite_map.get(self.snake_direction, -1) == action:
            raise ValueError(f"禁止反向移动: {self.snake_direction} -> {action}")

    def __del__(self) -> None:
        """析构函数确保资源释放"""
        self.close()