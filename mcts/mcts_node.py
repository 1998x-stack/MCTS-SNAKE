"""
mcts_node.py
MCTS节点实现
"""

import numpy as np
from typing import Optional, Dict, List, Any
from loguru import logger

class MCTSNode:
    """MCTS节点实现 (内存优化)"""
    
    __slots__ = ('parent', 'action', 'children', 'visit_count', 'total_value', 
                 'prior_prob', 'state_hash', 'is_terminal', 'reward')
    
    def __init__(self, 
                 state_hash: np.ndarray, 
                 parent: Optional['MCTSNode'] = None,
                 action: Optional[int] = None) -> None:
        """初始化节点
        
        Args:
            state_hash: 当前状态哈希 (状态压缩)
            parent: 父节点引用
            action: 导致当前状态的action
        """
        self.parent = parent
        self.action = action
        self.children: Dict[int, 'MCTSNode'] = {}  # 子节点字典 (action为key)
        
        # 统计信息
        self.visit_count = 0
        self.total_value = 0.0
        self.prior_prob = 0.0  # 先验概率
        
        # 状态信息
        self.state_hash = state_hash
        self.is_terminal = False
        self.reward = 0.0
        
        logger.debug(f"创建新节点 | 动作: {action} | 父节点: {id(parent)}")

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
        if not self.children:
            logger.warning("没有子节点可供选择")
            return self
            
        return max(self.children.values(), key=lambda x: x.visit_count)