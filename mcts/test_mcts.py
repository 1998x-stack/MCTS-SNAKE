import unittest, io, logging
import numpy as np
from snake_env import SnakeEnv
from mcts_controller import MCTSNode, MCTS

class TestMCTS(unittest.TestCase):
    """MCTS 控制器测试套件"""
    
    def setUp(self) -> None:
        """测试环境初始化"""
        self.env = SnakeEnv(grid_size=10)
        self.env.reset()  # 新增关键代码
        self.mcts = MCTS(self.env, simulation_budget=100)
        self.sim_env = SnakeEnv(grid_size=10)
        self.sim_env.reset()

    def test_node_initialization(self) -> None:
        """测试MCTS节点初始化"""
        state = self.mcts._state_hash(self.env)
        node = MCTSNode(state=state)
        
        self.assertEqual(node.visit_count, 0, "节点访问次数初始化错误")
        self.assertEqual(node.total_value, 0.0, "节点总价值初始化错误")
        self.assertIsNone(node.parent, "父节点引用未正确初始化")

    def test_uct_calculation(self) -> None:
        """测试UCT分数计算逻辑"""
        parent = MCTSNode(state=np.zeros((10,10,3)))
        parent.visit_count = 10
        
        child = MCTSNode(state=np.ones((10,10,3)), parent=parent)
        child.visit_count = 5
        child.total_value = 3.0
        
        expected_uct = (3/5) + 1.414 * np.sqrt(np.log(10)/5)
        self.assertAlmostEqual(child.uct_score(), expected_uct, delta=1e-3,
                              msg="UCT计算不符合公式")

    def test_selection_phase(self) -> None:
        """测试选择阶段逻辑"""
        # 构建测试树结构
        root = MCTSNode(state=np.zeros((10,10,3)))
        for action in [0,1,2]:
            child = MCTSNode(state=np.ones((10,10,3)), parent=root)
            child.visit_count = action+1
            child.total_value = (action+1)*0.5
            root.children[action] = child
        
        # 验证选择最优子节点
        selected = self.mcts._select_child(root)
        self.assertEqual(selected.uct_score(), max(c.uct_score() for c in root.children.values()),
                        "UCT选择策略失效")

    def test_expansion_phase(self) -> None:
        """测试节点扩展逻辑"""
        root = MCTSNode(state=self.mcts._state_hash(self.env))
        expanded_node = self.mcts._expand(root)
        
        self.assertEqual(len(root.children), 1, "子节点数量错误")
        self.assertIn(expanded_node.action, [0,1,2,3], "扩展动作非法")
        self.assertEqual(expanded_node.parent, root, "父节点引用未正确设置")

    def test_simulation_policy_mix(self) -> None:
        """测试混合模拟策略有效性"""
        # 设置特定测试场景
        self.sim_env.snake_body = [(5,5)]
        self.sim_env.food_pos = (5,7)  # 右侧有食物
        
        # 验证启发式策略选择向右移动
        action = self.mcts._heuristic_policy(self.sim_env)
        self.assertEqual(action, 3, "启发式策略方向选择错误")

    def test_backpropagation_flow(self) -> None:
        """测试反向传播数据流"""
        # 构建三级节点链
        root = MCTSNode(state=np.zeros((10,10,3)))
        child = MCTSNode(state=np.ones((10,10,3)), parent=root)
        grandchild = MCTSNode(state=np.ones((10,10,3)), parent=child)
        
        # 执行反向传播
        self.mcts._backpropagate(grandchild, 2.5)
        
        # 验证所有节点更新
        self.assertEqual(root.visit_count, 1, "根节点访问次数未更新")
        self.assertEqual(child.total_value, 2.5, "中间节点价值累积错误")
        self.assertEqual(grandchild.visit_count, 1, "叶子节点访问次数未更新")

    def test_terminal_node_handling(self) -> None:
        """测试终端节点处理"""
        # 创建终止节点
        terminal_node = MCTSNode(state=np.zeros((10,10,3)))
        terminal_node._terminal = True
        
        # 验证不会扩展终止节点
        result = self.mcts._expand(terminal_node)
        self.assertEqual(result, terminal_node, "错误扩展终止节点")

    # def test_state_hashing_consistency(self) -> None:
    #     """测试状态哈希一致性"""
    #     # 固定所有随机源
    #     seed = 42
    #     self.env.seed(seed)
    #     self.env.action_space.seed(seed)
    #     np.random.seed(seed)
        
    #     # 获取基准哈希
    #     self.env.reset()
    #     hash1 = self.mcts._state_hash(self.env)
        
    #     # 完全重置环境
    #     self.env.seed(seed)
    #     self.env.reset()
    #     hash2 = self.mcts._state_hash(self.env)
        
    #     # 允许浮点数微小误差
    #     np.testing.assert_allclose(hash1, hash2, rtol=1e-6, err_msg="核心特征编码不一致")

    def test_search_process_integration(self) -> None:
        """测试完整搜索流程集成"""
        best_action = self.mcts.search()
        self.assertIn(best_action, [0,1,2,3], "返回非法动作")
        
        # 验证根节点更新
        self.assertGreater(self.mcts.root.visit_count, 0, "搜索过程未更新节点")

    def test_performance_monitoring(self) -> None:
        """测试性能监控系统"""
        # 添加临时日志处理器
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        self.mcts._logger.addHandler(handler)
        
        # 执行基准搜索
        self.mcts.search()
        
        # 验证日志内容
        log_content = log_stream.getvalue()
        self.assertIn("Simulation completed", log_content)
        
        # 清理处理器
        self.mcts._logger.removeHandler(handler)

if __name__ == "__main__":
    unittest.main(
        verbosity=2,
        failfast=True,
        testRunner=unittest.TextTestRunner(
            descriptions="MCTS 测试报告"
        )
    )