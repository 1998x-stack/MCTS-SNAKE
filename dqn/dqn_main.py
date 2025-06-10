"""
dqn_main.py
Double Dueling DQN主训练脚本
"""

import time
import numpy as np
from tqdm import tqdm
import torch
from loguru import logger
from snake_env import SnakeEnv
from dqn_agent import DQNAgent, DQNConfig
from config import DQNConfig as Config

def preprocess_state(state: np.ndarray, frame_stack: np.ndarray) -> np.ndarray:
    """预处理状态并堆叠帧"""
    # 归一化到[0,1]范围
    state = state.astype(np.float32) / 255.0
    
    # 通道优先 (CHW格式)
    state = np.transpose(state, (2, 0, 1))
    
    # 帧堆叠
    if frame_stack is None:
        frame_stack = np.tile(state, (4, 1, 1))
    else:
        frame_stack = np.roll(frame_stack, -1, axis=0)
        frame_stack[-1] = state
    
    return frame_stack

def train_dqn(config: DQNConfig):
    """训练DQN智能体"""
    # 初始化环境
    env = SnakeEnv(grid_size=config.GRID_SIZE)
    n_actions = env.action_space.n
    
    # 计算状态形状
    input_shape = (config.FRAME_STACK, config.GRID_SIZE, config.GRID_SIZE)
    
    # 初始化智能体
    agent = DQNAgent(input_shape, n_actions, config)
    
    # 训练统计
    best_score = -np.inf
    scores = []
    
    logger.info("开始训练...")
    
    # 进度条
    progress = tqdm(total=config.TRAIN_STEPS, desc="训练进度")
    
    # 帧堆叠缓冲区
    frame_stack = None
    
    # 主训练循环
    while agent.steps_done < config.TRAIN_STEPS:
        # 重置环境
        state = env.reset()
        frame_stack = preprocess_state(state, None)
        
        done = False
        episode_reward = 0
        
        while not done:
            # 选择动作
            action = agent.select_action(frame_stack)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 预处理下一状态
            next_frame_stack = preprocess_state(next_state, frame_stack.copy())
            
            # 存储经验
            agent.push_memory(frame_stack, action, reward, next_frame_stack, done)
            
            # 优化模型
            loss = agent.optimize_model()
            
            # 更新目标网络
            agent.update_target_net()
            
            # 更新探索率
            agent.update_epsilon()
            
            # 更新状态
            frame_stack = next_frame_stack
            episode_reward += reward
            
            # 记录训练步
            if agent.steps_done % config.LOG_FREQ == 0:
                agent.log_step(reward)
            
            # 定期保存模型
            if agent.steps_done % config.SAVE_FREQ == 0:
                agent.save_model(f"snake_dqn_{agent.steps_done}.pth")
            
            # 定期评估
            if agent.steps_done % config.EVAL_FREQ == 0:
                score = evaluate(agent, env, config)
                scores.append(score)
                
                # 更新最佳模型
                if score > best_score:
                    best_score = score
                    agent.save_model("snake_dqn_best.pth")
                    logger.success(f"新最佳分数: {best_score} | 步数: {agent.steps_done}")
            
            progress.update(1)
            
            if agent.steps_done >= config.TRAIN_STEPS:
                break
            
        # 记录回合奖励
        agent.writer.add_scalar('Episode/Reward', episode_reward, agent.steps_done)
        logger.info(f"回合结束 | 总奖励: {episode_reward:.2f} | 蛇长: {len(env.snake_body)}")
    
    # 训练结束
    agent.save_model("snake_dqn_final.pth")
    agent.close()
    env.close()
    logger.success("训练完成!")
    
    # 显示最终结果
    final_score = evaluate(agent, env, config, render=True)
    logger.success(f"最终评估分数: {final_score}")

def evaluate(agent: DQNAgent, env: SnakeEnv, config: DQNConfig, render: bool = False) -> float:
    """评估智能体性能"""
    scores = []
    logger.info("评估智能体...")
    
    # 切换到评估模式
    agent.policy_net.eval()
    
    for _ in range(config.EVAL_EPISODES):
        state = env.reset()
        frame_stack = preprocess_state(state, None)
        
        done = False
        episode_score = 0
        
        while not done:
            if render:
                env.render()
                time.sleep(0.05)
            
            # 在评估模式下使用贪婪策略
            state_tensor = torch.tensor(frame_stack, dtype=torch.float32).unsqueeze(0).to(agent.device)
            action = agent.policy_net.act(state_tensor)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 预处理下一状态
            frame_stack = preprocess_state(next_state, frame_stack.copy())
            
            episode_score = len(env.snake_body) - 1
        
        scores.append(episode_score)
        logger.debug(f"评估回合 | 分数: {episode_score}")
    
    # 恢复训练模式
    agent.policy_net.train()
    return np.mean(scores)

if __name__ == "__main__":
    # 加载配置
    config = Config()
    config.display()
    
    # 开始训练
    try:
        train_dqn(config)
    except KeyboardInterrupt:
        logger.warning("训练被用户中断!")
    except Exception as e:
        logger.error(f"训练发生错误: {e}")
        raise e