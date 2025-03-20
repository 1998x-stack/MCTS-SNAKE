"""
mcts_snake_main.py
工业级MCTS贪吃蛇训练与可视化主程序
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict
from snake_env import SnakeEnv
from mcts_controller import MCTS

def main():
    # 配置命令行参数
    parser = argparse.ArgumentParser(description='MCTS贪吃蛇训练器')
    parser.add_argument('--grid_size', type=int, default=10,
                       help='游戏网格尺寸 (默认: 10)')
    parser.add_argument('--sim_budget', type=int, default=100,
                       help='MCTS每次决策的模拟次数 (默认: 100)')
    parser.add_argument('--max_steps', type=int, default=500,
                       help='单次游戏最大步数 (默认: 500)')
    args = parser.parse_args()

    # 初始化训练系统
    env = SnakeEnv(grid_size=args.grid_size)
    mcts = MCTS(env, simulation_budget=args.sim_budget)
    
    # 训练数据记录器
    training_data: List[Dict[str, float]] = []
    cumulative_scores: List[int] = []
    
    # 训练进度监控
    progress_bar = tqdm(range(1000), desc="训练周期", unit="epoch")

    # 主训练循环
    for epoch in progress_bar:
        # 环境初始化
        obs = env.reset()
        total_reward = 0.0
        step_count = 0
        done = False
        
        # 单次游戏循环
        while not done and step_count < args.max_steps:
            # MCTS决策
            best_action = mcts.search()
            
            # 执行动作
            obs, reward, done, info = env.step(best_action)
            
            # 更新状态
            mcts.update_root(best_action)
            
            # 记录数据
            total_reward += reward
            step_count += 1
            
            # 实时显示（每50步更新一次）
            if step_count % 50 == 0:
                env.render(mode='human')
                time.sleep(0.1)
        
        # 记录本周期结果
        epoch_data = {
            'epoch': epoch + 1,
            'total_score': len(env.snake_body) - 1,  # 初始长度为1
            'steps': step_count,
            'avg_reward': total_reward / step_count if step_count > 0 else 0
        }
        training_data.append(epoch_data)
        cumulative_scores.append(epoch_data['total_score'])
        
        # 更新进度条显示
        progress_bar.set_postfix({
            '当前分数': epoch_data['total_score'],
            '平均分数': np.mean([d['total_score'] for d in training_data[-50:]])
        })

    # 训练结束关闭环境
    env.close()

    # 可视化分析系统
    plt.figure(figsize=(12, 6))
    
    # 累计分数曲线
    plt.subplot(1, 2, 1)
    plt.plot(cumulative_scores, color='royalblue')
    plt.title('累计分数趋势分析')
    plt.xlabel('训练周期')
    plt.ylabel('游戏分数')
    plt.grid(True, alpha=0.3)
    
    # 移动平均曲线
    window_size = 50
    moving_avg = np.convolve(
        cumulative_scores, 
        np.ones(window_size)/window_size, 
        mode='valid'
    )
    plt.subplot(1, 2, 2)
    plt.plot(moving_avg, color='darkorange')
    plt.title(f'{window_size}周期移动平均')
    plt.xlabel('训练周期')
    plt.ylabel('平均分数')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存训练报告
    report_path = 'mcts_snake_training_report.png'
    plt.savefig(report_path, dpi=300)
    print(f"\n训练报告已保存至: {report_path}")

    # 关键指标分析
    final_performance = {
        '最大分数': np.max(cumulative_scores),
        '平均分数': np.mean(cumulative_scores),
        '分数标准差': np.std(cumulative_scores),
        '存活步数': np.mean([d['steps'] for d in training_data])
    }
    
    print("\n===== 训练结果分析 =====")
    for k, v in final_performance.items():
        print(f"{k}: {v:.2f}")

if __name__ == "__main__":
    # 启动性能监控
    start_time = time.time()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练过程被用户中断!")
    finally:
        print(f"总训练时长: {time.time()-start_time:.2f}秒")