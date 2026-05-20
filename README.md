# MASDiff 框架（仅主流程 + 抽象接口）


本目录下的 `src/` 是一个**框架工程**：只实现你在 `主流程.txt` 里定义的**固定主流程**与相关**抽象接口（base 类）**，便于用户接入：

- 自定义环境（仿真与数据收集）
- 自定义 DQN（策略初始化、训练、数据构建）
- 自定义扩散模型（随机初始化、按 Tau 条件生成奖励、截断扩散变异、按种群训练）
- 自定义进化策略（精英选择等）

框架不提供任何具体环境 / DQN / 扩散模型 / 进化策略实现；你需要在自己的工程或本工程外部模块中实现这些接口，并在 YAML 配置中通过 Python 导入路径接入。

## 快速开始

1. 安装依赖（仅用于读 YAML）：

```bash
pip install -r requirements.txt
```

2. 复制并修改示例配置：

- 示例：`configs/default.yaml`

3. 运行（需要你提供各模块的具体实现类）：

```bash
python run.py --config configs/default.yaml
```

## 代码结构

- `src/config/`：配置 dataclass 与 YAML 加载
- `src/pipeline/`：主流程 runner 与 steps（每一步一个函数）
- `src/environments/`：环境接口
- `src/dqn/`：DQN 模块接口
- `src/diffusion/`：扩散模型接口
- `src/evolution/`：进化（精英选择等）接口
- `src/q/`：Q（目标/标签）加载或创建接口
- `src/parallel/`：并行执行接口（默认串行）
- `configs/`：每套模块配置一个 YAML（示例为 `default.yaml`）

