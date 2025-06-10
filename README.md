# 🐍 工业级贪吃蛇 AI：基于 MCTS 的智能决策系统

> 🚀 一个用于研究与部署的高性能贪吃蛇 AI 解决方案，采用蒙特卡洛树搜索（MCTS）驱动智能决策，具备可视化监控、可扩展训练与性能分析能力。

---

## 📌 项目简介

本项目实现了一个 **工业级别的贪吃蛇智能体训练系统**，核心算法基于 Monte Carlo Tree Search（MCTS），可用于教学演示、科研探索与真实部署任务。项目内包含：

* ✅ 自定义 Gym 环境模拟器
* ✅ 高效多线程 MCTS 决策器
* ✅ TensorboardX 可视化训练监控
* ✅ 精细化性能分析与日志系统
* ✅ 强健的错误处理与模块化设计

---

## 🚀 核心特性

* 🧠 **自定义贪吃蛇环境**（兼容 OpenAI Gym 接口）
* 🌀 **高效 MCTS 控制器**，支持并行模拟与缓存优化
* 📊 **集成 TensorboardX**，实时可视化训练过程
* 📈 **实时性能监控**与最终报告自动生成
* 🧾 **结构化日志记录**（基于 Loguru）
* 🧪 **工业级代码质量**，强类型注解 + 规范 Docstring

---

## 🔧 安装依赖

```bash
pip install -r requirements.txt
```

---

## 🗂️ 项目结构

```bash
mcts-snake/
├── snake_env.py         # 贪吃蛇 Gym 环境
├── mcts_node.py         # MCTS 节点定义与统计信息
├── mcts_controller.py   # MCTS 决策主逻辑
├── mcts_snake_main.py   # 主训练脚本
├── requirements.txt     # 依赖文件
└── README.md            # 项目说明文档
```

---

## ⚡ 快速开始

### 默认训练启动：

```bash
python mcts_snake_main.py
```

### 自定义参数运行示例：

```bash
python mcts_snake_main.py \
    --grid_size 12 \
    --sim_budget 200 \
    --max_steps 1000 \
    --num_episodes 500
```

---

## 🛠️ 参数说明

| 参数名              | 默认值  | 含义说明            |
| ---------------- | ---- | --------------- |
| `--grid_size`    | 10   | 贪吃蛇游戏网格大小       |
| `--sim_budget`   | 100  | 每次 MCTS 决策的模拟次数 |
| `--max_steps`    | 500  | 单局游戏的最大步数       |
| `--num_episodes` | 1000 | 总训练轮次           |

---

## 📉 训练监控与可视化

* 实时控制台输出当前分数、平均分与最高分
* 使用 Tensorboard 可视化训练数据：

```bash
tensorboard --logdir logs
```

---

## 📤 训练输出内容

训练完成后自动生成：

* ✅ 控制台打印关键训练指标
* ✅ PNG 格式训练报告图表
* ✅ 完整日志文件（`snake_mcts.log`）

---

## 🧩 核心算法说明

### 贪吃蛇环境设计

* 状态表示：3 通道张量（蛇体、食物、方向）
* 奖励函数：吃食+、距离缩短+、撞墙/自撞-
* 碰撞检测：自身体积碰撞逻辑实现

### MCTS 决策系统

* **选择（Selection）**：UCT 策略选择最优节点
* **扩展（Expansion）**：添加未探索子节点
* **模拟（Simulation）**：启发式模拟未来状态
* **回传（Backpropagation）**：更新节点 Q 值与访问次数

---

## ⚙️ 性能优化机制

* ✅ 状态哈希缓存加速重复状态检索
* ✅ 多线程并行模拟支持
* ✅ 批处理 MCTS 节点处理机制

---

## 🔧 可扩展性与自定义

* **修改环境规则**：编辑 `snake_env.py` 中奖励或状态逻辑
* **调整搜索策略**：在 `mcts_controller.py` 中更改探索参数或模拟策略
* **引入神经网络策略**：可无缝集成 Policy Network 实现 Hybrid-MCTS 架构

---

## 📈 性能评估指标

训练结束后输出以下核心指标：

* 最高分
* 平均得分
* 分数标准差
* 平均存活步数
* 学习效率估计

---

## 🤝 贡献指南

我们欢迎任何贡献，包括优化算法、改进可视化、重构模块等，请遵循以下规范：

* [PEP 8](https://peps.python.org/pep-0008/) 编码规范
* Google Python 风格指南
* 全面使用 Python 类型注解
* 所有模块必须附带完整 Docstring

---

## 📄 许可证

本项目遵循 `LICENSE` 协议开源，使用请遵守许可协议内容。

---

## 📬 联系我们

如有任何建议或反馈，欢迎通过 Issues 提交或直接联系我们的维护团队！