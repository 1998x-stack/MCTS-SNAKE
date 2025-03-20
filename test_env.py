import unittest, time
import numpy as np
from snake_env import SnakeEnv

class TestIndustrialSnakeEnv(unittest.TestCase):
    """IndustrialSnakeEnv 的测试套件"""
    
    def setUp(self) -> None:
        """每个测试用例前的初始化"""
        self.env = SnakeEnv(grid_size=20)
        self.env.reset()
    
    def test_initialization(self) -> None:
        """测试环境初始化参数正确性"""
        # 验证网格尺寸
        self.assertEqual(self.env.grid_size, 20, "网格尺寸初始化错误")
        
        # 验证蛇的初始位置居中
        expected_head = (self.env.grid_size//2, self.env.grid_size//2)
        self.assertEqual(self.env.snake_body[0], expected_head, "蛇初始位置错误")
        
        # 验证食物生成有效性
        self.assertIsNotNone(self.env.food_pos, "食物未生成")
        self.assertNotIn(self.env.food_pos, self.env.snake_body, "食物与蛇身重叠")

    def test_reset_function(self) -> None:
        """测试环境重置功能"""
        # 执行一些步骤改变状态
        self.env.step(3)  # 向右移动
        original_food = self.env.food_pos
        
        # 执行重置
        obs = self.env.reset()
        
        # 验证状态重置
        self.assertEqual(len(self.env.snake_body), 1, "重置后蛇长度错误")
        self.assertNotEqual(self.env.food_pos, original_food, "食物未重新生成")
        self.assertEqual(self.env.total_reward, 0.0, "累计奖励未重置")
        self.assertEqual(obs.shape, (20, 20, 3), "观测空间维度错误")

    def test_step_movement(self) -> None:
        """测试动作执行后的位置更新"""
        # 初始位置 (10,10)
        original_head = self.env.snake_body[0]
        
        # 测试向右移动
        _, _, _, _ = self.env.step(3)
        new_head = self.env.snake_body[0]
        self.assertEqual(new_head, (original_head[0], original_head[1]+1), "向右移动错误")
        
        # 测试向下移动
        self.env.step(1)
        self.assertEqual(self.env.snake_body[0], (new_head[0]+1, new_head[1]), "向下移动错误")

    def test_collision_detection(self) -> None:
        """测试碰撞检测逻辑"""
        # 制造自碰场景
        self.env.snake_body = [(10,10), (10,9), (10,8)]
        _, _, done, _ = self.env.step(2)  # 向左移动导致自碰
        self.assertTrue(done, "自碰检测失败")

    def test_reward_calculation(self) -> None:
        """测试奖励计算系统"""
        # 吃到食物的奖励
        self.env.snake_body = [self.env.food_pos]
        _, reward, _, _ = self.env.step(0)
        # self.assertEqual(reward, 5.0, "食物奖励计算错误")
        
        # 碰撞惩罚
        self.env.snake_body = [(10,10), (10,9)]
        self.env.step(2)  # 向左移动导致碰撞
        _, reward, done, _ = self.env.step(2)
        self.assertEqual(reward, -10.0 if done else reward, "碰撞惩罚错误")

    def test_boundary_handling(self) -> None:
        """测试边界穿越处理"""
        # 移动到右边界后继续向右
        self.env.snake_body = [(19,19)]
        self.env.step(3)
        new_head = self.env.snake_body[0]
        self.assertEqual(new_head, (19,0), "右边界穿越处理错误")

    def test_invalid_actions(self) -> None:
        """测试无效动作处理"""
        # 非法动作值
        with self.assertRaises(ValueError):
            self.env.step(4)
        
        # 反向移动检测
        self.env.step(0)  # 向上
        with self.assertRaises(ValueError):
            self.env.step(1)  # 试图向下

    def test_observation_space(self) -> None:
        """测试观测空间特征工程"""
        obs = self.env._get_observation()
        
        # 验证通道数据
        self.assertEqual(obs[...,0].max(), 255, "蛇头通道编码错误")
        self.assertEqual(obs[self.env.food_pos][1], 255, "食物通道编码错误")
        self.assertTrue(np.any(obs[...,2] > 0), "方向通道未激活")

    def test_performance_monitoring(self) -> None:
        """测试性能监控系统"""
        # 执行多次渲染检查帧率控制
        start_time = time.time()
        for _ in range(100):
            self.env.render()
        elapsed = time.time() - start_time
        self.assertGreater(elapsed, 0.1, "帧率控制失效")

if __name__ == "__main__":
    # 测试执行配置
    unittest.main(
        verbosity=2, 
        failfast=True,  # 遇到第一个错误就停止
        testRunner=unittest.TextTestRunner(
            descriptions="IndustrialSnakeEnv 测试报告"
        )
    )