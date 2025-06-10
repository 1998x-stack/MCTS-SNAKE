"""
mcts_controller.py
MCTS控制器实现
"""

import numpy as np
import time
import logging
from typing import Dict, Optional, Tuple, Any, List
from snake_env import SnakeEnv
from mcts_node import MCTSNode
from loguru import logger
import concurrent.futures
from functools import lru_cache
import copy

class MCTSController:
    """MCTS控制器"""
    
    def __init__(self, 
                 env: SnakeEnv,
                 simulation_budget: int = 1000,
                 num_workers: int = 4,
                 exploration_factor: float = 1.414) -> None:
        """初始化控制器
        
        Args:
            env: Snake环境实例
            simulation_budget: 每次决策的模拟次数
            num_workers: 并行工作进程数量
            exploration_factor: UCT探索因子
        """
        self.env = env
        self.simulation_budget = simulation_budget
        self.num_workers = min(num_workers, simulation_budget)
        self.exploration_factor = exploration_factor
        
        # 初始化根节点
        self.root = MCTSNode(self._state_hash())
        
        # 性能监控
        self.total_simulations = 0
        self.simulation_time = 0.0
        logger.info(f"MCTS控制器初始化 | 模拟预算: {simulation_budget} | 工作线程: {num_workers}")

    def search(self) -> int:
        """执行完整MCTS搜索流程"""
        start_time = time.time()
        
        # 创建并行任务
        tasks = []
        for _ in range(self.simulation_budget):
            # 保存当前环境状态
            env_state = self._save_env_state()
            # 添加模拟任务
            tasks.append((copy.deepcopy(self.root), env_state))
        
        # 并行执行模拟
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_task = {executor.submit(self._simulate, task[0], task[1]): task for task in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                results.append(future.result())
        
        # 合并结果并反向传播
        for node, cumulative_reward in results:
            self._backpropagate(node, cumulative_reward)
        
        self.simulation_time = time.time() - start_time
        self.total_simulations += self.simulation_budget
        
        logger.info(f"MCTS搜索完成 | 耗时: {self.simulation_time:.2f}s | 平均模拟时间: {self.simulation_time/self.simulation_budget*1000:.2f}ms")
        
        # 返回最佳动作
        best_child = self.root.best_child()
        return best_child.action if best_child.action is not None else 0

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """选择阶段 (UCT改进)"""
        if not node.children:
            return node
            
        return max(node.children.values(), key=lambda x: x.uct_score(self.exploration_factor))

    def _expand(self, node: MCTSNode, env_state: Dict[str, Any]) -> MCTSNode:
        """扩展阶段"""
        # 检查是否为终止状态
        if node.is_terminal:
            return node
        
        # 选择未尝试的动作
        tried_actions = set(node.children.keys())
        valid_actions = [a for a in range(4) if a not in tried_actions]
        
        if not valid_actions:
            return node
            
        # 选择动作
        action = np.random.choice(valid_actions)
        
        # 创建新环境副本执行动作
        env_copy = SnakeEnv(grid_size=self.env.grid_size)
        self._restore_env_state(env_copy, env_state)
        try:
            _, _, done, _ = env_copy.step(action)
        except Exception as e:
            logger.error(f"扩展节点时出错: {e}")
            done = True
            
        # 创建新节点
        new_node = MCTSNode(
            state_hash=self._state_hash(env=env_copy),
            parent=node,
            action=action
        )
        new_node.is_terminal = done
        new_node.reward = env_copy.total_reward
        
        # 添加到子节点
        node.children[action] = new_node
        return new_node

    def _simulate(self, node: MCTSNode, env_state: Dict[str, Any]) -> Tuple[MCTSNode, float]:
        """模拟阶段 (策略优化)"""
        # 恢复环境状态
        env_copy = SnakeEnv(grid_size=self.env.grid_size)
        self._restore_env_state(env_copy, env_state)
        
        total_reward = 0.0
        steps = 0
        current_node = node
        max_steps = 100  # 防止无限循环
        
        # 选择与扩展循环
        while not current_node.is_terminal and steps < max_steps:
            if not current_node.is_fully_expanded():
                current_node = self._expand(current_node, self._save_env_state(env=env_copy))
                # 新节点可能是终止状态
                if current_node.is_terminal:
                    total_reward += current_node.reward
                    break
            else:
                current_node = self._select_child(current_node)
                
            # 在环境中执行动作
            try:
                # 如果当前节点是新建的，已经有动作
                if current_node.action is not None:
                    _, reward, done, _ = env_copy.step(current_node.action)
                    total_reward += reward
                    steps += 1
                    if done:
                        current_node.is_terminal = True
                        current_node.reward = reward
                        break
            except Exception as e:
                logger.error(f"模拟时执行动作出错: {e}")
                current_node.is_terminal = True
                break
        
        logger.debug(f"模拟完成 | 总奖励: {total_reward:.2f} | 步数: {steps}")
        return current_node, total_reward

    def _heuristic_policy(self, env: SnakeEnv) -> int:
        """启发式策略 (向量化)"""
        if not env.snake_body or env.food_pos is None:
            return 0
            
        head = np.array(env.snake_body[0])
        food = np.array(env.food_pos)
        
        # 计算距离差
        diff = food - head
        
        # 根据距离选择最佳方向
        if abs(diff[0]) > abs(diff[1]):
            return 1 if diff[0] > 0 else 0  # 下或上
        else:
            return 3 if diff[1] > 0 else 2  # 右或左

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """反向传播 (优化版本)"""
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            value += current.reward if current.reward is not None else 0
            current = current.parent

    def _save_env_state(self, env: Optional[SnakeEnv] = None) -> Dict[str, Any]:
        """保存环境状态 (轻量级版本)"""
        if env is None:
            env = self.env
            
        return {
            'snake_body': [tuple(pos) for pos in env.snake_body],
            'food_pos': tuple(env.food_pos) if env.food_pos is not None else (0, 0),
            'snake_direction': env.snake_direction,
            'total_reward': env.total_reward,
            'step_counter': env.step_counter
        }
    
    def _restore_env_state(self, env: SnakeEnv, state: dict) -> None:
        """恢复环境状态"""
        env.snake_body = [list(pos) for pos in state['snake_body']]
        env.food_pos = list(state['food_pos'])
        env.snake_direction = state['snake_direction']
        env.total_reward = state['total_reward']
        env.step_counter = state['step_counter']

    def _state_hash(self, env: Optional[SnakeEnv] = None) -> np.ndarray:
        """生成状态哈希 (向量化)"""
        if env is None:
            env = self.env
        
        # 基础特征向量
        features = np.zeros(5, dtype=np.float32)
        
        if not env.snake_body or env.food_pos is None:
            return features
            
        # 蛇头位置 (归一化)
        head = np.array(env.snake_body[0], dtype=np.float32)
        features[:2] = head / env.grid_size
        
        # 食物相对位置
        food = np.array(env.food_pos, dtype=np.float32)
        features[2:4] = (food - head) / env.grid_size
        
        # 方向编码
        features[4] = env.snake_direction / 4.0
        
        return features

    def update_root(self, action: int) -> None:
        """更新根节点 (状态管理)"""
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            # 环境可能已发生变化，重新创建根节点
            self.root = MCTSNode(self._state_hash())
            
        logger.debug(f"更新根节点 | 新动作: {action}")