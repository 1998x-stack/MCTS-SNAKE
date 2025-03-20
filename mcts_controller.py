import numpy as np
from typing import Dict, Optional, Tuple, List
import copy
import logging
from snake_env import SnakeEnv
import concurrent.futures
from functools import lru_cache


class MCTSNode:
    """MCTS节点实现 (内存优化)"""
    
    __slots__ = ('parent', 'action', 'children', 'visit_count', 'total_value', 
                 'prior_prob', '_state', '_terminal', '_reward', '_best_action_cache')
    
    def __init__(self, 
                 state: np.ndarray, 
                 parent: Optional['MCTSNode'] = None,
                 action: Optional[int] = None) -> None:
        """初始化节点
        
        Args:
            state: 当前状态哈希 (状态压缩)
            parent: 父节点引用
            action: 导致当前状态的action
        """
        self.parent = parent
        self.action = action
        self.children: Dict[int, MCTSNode] = {}  # 子节点字典 (action为key)
        
        # 统计信息
        self.visit_count = 0
        self.total_value = 0.0
        self.prior_prob = 0.0  # 先验概率 (改进)
        
        # 状态缓存
        self._state = state
        self._terminal = False
        self._reward = 0.0
        
        # 缓存最佳动作
        self._best_action_cache = None

    def uct_score(self, exploration_factor: float = 1.414) -> float:
        """计算UCT分数"""
        if self.visit_count == 0:
            return float('inf')
        
        parent_visits = self.parent.visit_count if self.parent else 1
        log_parent = np.log(parent_visits) if parent_visits > 0 else 1
        
        exploitation = self.total_value / self.visit_count
        exploration = exploration_factor * np.sqrt(log_parent / self.visit_count)
        return exploitation + exploration + self.prior_prob

    def is_fully_expanded(self) -> bool:
        """检查是否完全扩展"""
        return len(self.children) == 4  # 4个可能动作

    def best_child(self) -> 'MCTSNode':
        """选择最优子节点 (策略)"""
        if self._best_action_cache is not None and self._best_action_cache in self.children:
            return self.children[self._best_action_cache]
        
        best_node = max(self.children.values(), key=lambda x: x.visit_count)
        self._best_action_cache = best_node.action
        return best_node


class MCTS:
    """MCTS控制器"""
    
    def __init__(self, 
                 env: SnakeEnv,
                 simulation_budget: int = 1000,
                 num_workers: int = 4) -> None:
        """初始化控制器
        
        Args:
            env: Snake环境实例
            simulation_budget: 每次决策的模拟次数
            num_workers: 并行工作进程数量
        """
        self.root = MCTSNode(self._state_hash(env))
        self.env = env
        self.simulation_budget = simulation_budget
        self.num_workers = num_workers
        
        # 性能监控
        self._logger = logging.getLogger('MCTS')
        if not self._logger.handlers:  # 防止重复添加
            self._logger.addHandler(logging.NullHandler())
            self._logger.setLevel(logging.INFO)
        self._simulation_time = 0.0
        
        # 缓存
        self._state_hash_cache = {}

    def search(self) -> int:
        """执行完整MCTS搜索流程"""
        # 使用批量并行模拟
        batch_size = max(1, self.simulation_budget // self.num_workers)
        for _ in range(0, self.simulation_budget, batch_size):
            # 准备当前批次的模拟任务
            simulation_tasks = []
            for _ in range(min(batch_size, self.simulation_budget - _)):
                # 保存当前环境状态 - 使用更高效的方式
                original_state = self._save_env_state(self.env)
                
                # 选择阶段
                node = self.root
                while not node._terminal and node.children and node.is_fully_expanded():
                    node = self._select_child(node)
                
                # 扩展阶段
                if not node._terminal and not node.is_fully_expanded():
                    node = self._expand(node)
                
                # 添加到模拟任务列表
                simulation_tasks.append((node, original_state))
                
            # 并行执行模拟
            if self.num_workers > 1:
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    results = list(executor.map(self._simulate_worker, simulation_tasks))
            else:
                results = [self._simulate_worker(task) for task in simulation_tasks]
            
            # 反向传播
            for node, reward in results:
                self._backpropagate(node, reward)
            
        return self.root.best_child().action

    def _save_env_state(self, env: SnakeEnv) -> dict:
        """保存环境状态 (轻量级版本)"""
        return {
            'snake_body': [tuple(pos) for pos in env.snake_body],
            'food_pos': tuple(env.food_pos),
            'snake_direction': env.snake_direction,
            'score': env.score,
            'grid_size': env.grid_size
        }
    
    def _restore_env_state(self, env: SnakeEnv, state: dict) -> None:
        """恢复环境状态"""
        env.snake_body = [list(pos) for pos in state['snake_body']]
        env.food_pos = list(state['food_pos'])
        env.snake_direction = state['snake_direction']
        env.score = state['score']
        env.grid_size = state['grid_size']

    def _simulate_worker(self, task_data: Tuple[MCTSNode, dict]) -> Tuple[MCTSNode, float]:
        """模拟工作函数 (用于并行处理)"""
        node, env_state = task_data
        env_copy = SnakeEnv()  # 创建新环境实例
        self._restore_env_state(env_copy, env_state)
        reward = self._simulate(env_copy)
        return node, reward

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """选择阶段 (UCT改进)"""
        return max(node.children.values(), key=lambda x: x.uct_score())

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """扩展阶段 (并行优化)"""
        
        # 添加终止状态检查
        if node._terminal:
            return node
        
        # 选择未尝试的动作
        tried_actions = set(node.children.keys())
        valid_actions = [a for a in range(4) if a not in tried_actions]
        
        # 向量化创建新节点
        if valid_actions:
            action = np.random.choice(valid_actions)
            new_node = MCTSNode(
                state=self._state_hash(self.env),
                parent=node,
                action=action
            )
            node.children[action] = new_node
            return new_node
        
        raise RuntimeError("无法扩展节点")

    @lru_cache(maxsize=1024)
    def _simulate(self, env: SnakeEnv) -> float:
        """模拟阶段 (策略优化 + 缓存)"""
        self._logger.info("Simulation started")
        
        # 预先计算可能的动作和相应的权重
        action_weights = np.ones(4) * 0.5  # 默认权重
        
        total_reward = 0.0
        steps = 0
        last_action = None
        
        while True:
            # 使用向量化操作选择动作
            if np.random.rand() < 0.5:
                action = self._heuristic_policy(env)
            else:
                # 向量化计算有效动作
                valid_mask = np.ones(4, dtype=bool)
                if last_action is not None:
                    opposite_action = {0:1, 1:0, 2:3, 3:2}.get(last_action)
                    valid_mask[opposite_action] = False
                
                valid_actions = np.arange(4)[valid_mask]
                if len(valid_actions) == 0:
                    action = 0
                else:
                    action = np.random.choice(valid_actions)
            
            try:
                _, reward, done, _ = env.step(action)
                last_action = action
            except ValueError:
                continue
                
            total_reward += reward
            steps += 1
        
            if done or steps > 100:
                break
        
        self._logger.info("Simulation completed with reward: %.2f", total_reward)
        return total_reward

    def _heuristic_policy(self, env: SnakeEnv) -> int:
        """启发式策略 (向量化)"""
        if not env.snake_body:
            return 0
            
        head = np.array(env.snake_body[0])
        food = np.array(env.food_pos)
        
        # 向量化计算相对位置
        diff = food - head
        
        # 使用向量化操作计算最佳动作
        action_priorities = np.zeros(4)
        
        # 优先处理x轴
        if diff[0] > 0:
            action_priorities[1] = 2  # 下
        elif diff[0] < 0:
            action_priorities[0] = 2  # 上
            
        # 其次处理y轴
        if diff[1] > 0:
            action_priorities[3] = 1  # 右
        elif diff[1] < 0:
            action_priorities[2] = 1  # 左
            
        # 筛选合法动作
        valid_actions = []
        try:
            valid_actions = [a for a in range(4) if a in env.action_space]
        except:
            valid_actions = list(range(4))
            
        if not valid_actions:
            return 0
            
        # 从优先级最高的合法动作中选择
        valid_priorities = action_priorities[valid_actions]
        best_indices = np.where(valid_priorities == np.max(valid_priorities))[0]
        best_action_idx = np.random.choice(best_indices)
        return valid_actions[best_action_idx]

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """反向传播 (优化版本)"""
        # 使用迭代而非递归
        nodes_to_update = []
        current = node
        
        # 收集需要更新的节点
        while current is not None:
            nodes_to_update.append(current)
            current = current.parent
            
        # 批量更新统计信息
        for n in nodes_to_update:
            n.visit_count += 1
            n.total_value += reward
            # 清除最佳动作缓存
            n._best_action_cache = None

    def _state_hash(self, env: SnakeEnv) -> np.ndarray:
        """生成状态哈希 (向量化 + 缓存)"""
        # 检查缓存
        env_id = id(env)
        if env_id in self._state_hash_cache:
            return self._state_hash_cache[env_id]
            
        if not env.snake_body:
            result = np.zeros(5, dtype=np.float32)
            self._state_hash_cache[env_id] = result
            return result
        
        # 向量化特征提取
        head = np.array(env.snake_body[0], dtype=np.float32)
        food = np.array(env.food_pos, dtype=np.float32)
        
        # 向量化计算
        normalized_head = head / env.grid_size
        relative_pos = (food - head) / 20.0
        direction_encoding = np.array([env.snake_direction / 4.0], dtype=np.float32)
        
        # 合并特征向量
        features = np.concatenate([normalized_head, relative_pos, direction_encoding])
        
        # 更新缓存
        self._state_hash_cache[env_id] = features
        return features

    def update_root(self, action: int) -> None:
        """更新根节点 (状态管理)"""
        # 清除状态哈希缓存
        self._state_hash_cache.clear()
        
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = MCTSNode(self._state_hash(self.env))