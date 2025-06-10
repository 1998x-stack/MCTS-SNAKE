"""
config.py
DQN训练参数配置
"""

class DQNConfig:
    """DQN训练参数配置"""
    def __init__(self):
        # 环境参数
        self.GRID_SIZE = 10                   # 游戏网格大小
        self.FRAME_STACK = 4                  # 状态帧堆叠数量
        
        # 网络架构
        self.DUELING = True                   # 使用Dueling DQN
        self.NOISY_NET = True                 # 使用Noisy Net
        
        # 训练参数
        self.GAMMA = 0.99                     # 折扣因子
        self.LR = 1e-4                        # 学习率
        self.BATCH_SIZE = 64                  # 批大小
        self.UPDATE_FREQ = 1000               # 目标网络更新频率
        self.REPLAY_SIZE = 10000              # 经验回放缓冲区大小
        self.START_EPSILON = 1.0              # 初始探索率
        self.END_EPSILON = 0.01               # 最终探索率
        self.EPSILON_DECAY = 10000            # 探索率衰减步数
        self.SAVE_FREQ = 1000                 # 模型保存频率
        self.LOG_FREQ = 100                   # 日志记录频率
        
        # 训练过程
        self.TRAIN_STEPS = 100000             # 总训练步数
        self.INITIAL_MEMORY = 1000            # 预热记忆大小
        
        # 评估参数
        self.EVAL_EPISODES = 10               # 评估周期数
        self.EVAL_FREQ = 5000                 # 评估频率

    def display(self):
        """打印配置信息"""
        print("="*50)
        print("DQN训练配置:")
        for key, value in vars(self).items():
            print(f"{key.replace('_', ' ').title():<20}: {value}")
        print("="*50)