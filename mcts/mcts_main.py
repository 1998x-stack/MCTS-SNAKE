"""
mcts_snake_main.py
工业级MCTS贪吃蛇训练与可视化主程序
"""

import argparse
import time
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from snake_env import SnakeEnv
from mcts_controller import MCTSController
from loguru import logger
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import os

# 配置日志
logger.add("logs/snake_mcts.log", rotation="100 MB", retention="10 days", level="DEBUG")

def train_mcts_snake(grid_size: int = 10, 
                     sim_budget: int = 100, 
                     max_steps: int = 500, 
                     num_episodes: int = 1000,
                     log_interval: int = 10) -> None:
    """训练MCTS贪吃蛇
    
    Args:
        grid_size: 游戏网格尺寸
        sim_budget: MCTS每次决策的模拟次数
        max_steps: 单次游戏最大步数
        num_episodes: 训练周期数
        log_interval: Tensorboard日志间隔
    """
    # 初始化环境、MCTS和Tensorboard
    env = SnakeEnv(grid_size=grid_size)
    mcts = MCTSController(env, simulation_budget=sim_budget)
    writer = SummaryWriter(f"logs/snake_{int(time.time())}")
    
    # 训练数据记录
    episode_scores = []
    episode_lengths = []
    episode_rewards = []
    moving_avg_scores = []
    best_score = 0
    
    logger.info("开始训练...")
    
    progress_bar = tqdm(range(num_episodes), desc="训练周期", unit="episode")
    
    for episode in progress_bar:
        # 环境初始化
        obs = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        
        # 单次游戏循环
        while not done and steps < max_steps:
            # MCTS决策
            best_action = mcts.search()
            
            # 执行动作
            obs, reward, done, info = env.step(best_action)
            
            # 更新MCTS根节点
            mcts.update_root(best_action)
            
            # 记录数据
            total_reward += reward
            steps += 1
            
            # Tensorboard记录
            if steps % log_interval == 0:
                writer.add_scalar('Reward/step_reward', reward, mcts.total_simulations)
                writer.add_scalar('Performance/snake_length', info['snake_length'], mcts.total_simulations)
        
        # 记录本周期结果
        score = len(env.snake_body) - 1  # 初始长度为1
        episode_scores.append(score)
        episode_lengths.append(steps)
        episode_rewards.append(total_reward)
        
        # 计算移动平均
        avg_score = np.mean(episode_scores[-50:])
        moving_avg_scores.append(avg_score)
        
        # 更新最佳分数
        if score > best_score:
            best_score = score
            logger.success(f"新最佳分数: {best_score} | 周期: {episode+1}")
        
        # Tensorboard记录
        writer.add_scalar('Score/episode_score', score, episode)
        writer.add_scalar('Reward/total_reward', total_reward, episode)
        writer.add_scalar('Performance/steps', steps, episode)
        writer.add_scalar('Performance/moving_avg_score', avg_score, episode)
        writer.add_scalar('MCTS/simulations_per_episode', mcts.simulation_budget * steps, episode)
        writer.add_scalar('MCTS/avg_simulation_time', mcts.simulation_time / steps, episode)
        
        # 更新进度条
        progress_bar.set_postfix({
            '当前分数': score,
            '平均分数': avg_score,
            '最佳分数': best_score
        })
    
    # 训练结束关闭环境
    env.close()
    writer.close()
    logger.info("训练完成")
    
    # 生成训练报告
    generate_training_report(episode_scores, moving_avg_scores)
    
    # 关键指标分析
    analyze_performance(episode_scores, episode_rewards, episode_lengths)

def generate_training_report(scores: List[int], moving_avg: List[float]) -> None:
    """生成训练报告可视化"""
    plt.figure(figsize=(12, 6))
    
    # 累计分数曲线
    plt.subplot(1, 2, 1)
    plt.plot(scores, color='royalblue')
    plt.title('累计分数趋势分析')
    plt.xlabel('训练周期')
    plt.ylabel('游戏分数')
    plt.grid(True, alpha=0.3)
    
    # 移动平均曲线
    plt.subplot(1, 2, 2)
    plt.plot(moving_avg, color='darkorange')
    plt.title('移动平均分数')
    plt.xlabel('训练周期')
    plt.ylabel('平均分数')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存训练报告
    os.makedirs("reports", exist_ok=True)
    report_path = 'reports/mcts_snake_training_report.png'
    plt.savefig(report_path, dpi=300)
    logger.success(f"训练报告已保存至: {report_path}")

def analyze_performance(scores: List[int], rewards: List[float], lengths: List[int]) -> None:
    """分析最终性能指标"""
    final_performance = {
        '最大分数': np.max(scores),
        '平均分数': np.mean(scores),
        '分数标准差': np.std(scores),
        '平均奖励': np.mean(rewards),
        '平均存活步数': np.mean(lengths),
        '学习效率': np.max(scores) / (np.mean(lengths) + 1e-5)
    }
    
    logger.info("\n===== 训练结果分析 =====")
    for k, v in final_performance.items():
        logger.info(f"{k}: {v:.2f}")

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='MCTS贪吃蛇训练器')
    parser.add_argument('--grid_size', type=int, default=10,
                       help='游戏网格尺寸 (默认: 10)')
    parser.add_argument('--sim_budget', type=int, default=100,
                       help='MCTS每次决策的模拟次数 (默认: 100)')
    parser.add_argument('--max_steps', type=int, default=500,
                       help='单次游戏最大步数 (默认: 500)')
    parser.add_argument('--num_episodes', type=int, default=1000,
                       help='训练周期数 (默认: 1000)')
    args = parser.parse_args()
    
    # 启动训练
    start_time = time.time()
    
    try:
        train_mcts_snake(
            grid_size=args.grid_size,
            sim_budget=args.sim_budget,
            max_steps=args.max_steps,
            num_episodes=args.num_episodes
        )
    except KeyboardInterrupt:
        logger.warning("训练过程被用户中断!")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
    finally:
        logger.info(f"总训练时长: {time.time()-start_time:.2f}秒")