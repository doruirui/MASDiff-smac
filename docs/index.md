# MASDiff 星际争霸（SMAC）场景文档

## 问题定义

在星际争霸多智能体战斗环境（SMAC）中，智能体（战斗单位）仅能获得稀疏的全局奖励（例如胜/负）。这导致独立学习的 DQN 策略难以收敛。本框架解决的核心问题是：

> **如何利用条件扩散模型，根据每个智能体的局部观测自动生成稠密奖励矩阵，从而有效训练每个单位的 DQN 策略，并通过种群进化持续优化奖励函数，最终提升战斗胜率？**

**形式化描述：**
- 环境包含 \( N \) 个友方单位，每个单位有动作空间 \( A \)（移动、攻击等）。
- 从仿真中收集条件信息 \( \tau \in \mathbb{R}^{N \times d} \)，\( \tau_i \) 为单位 i 的观测（血量、位置、敌我距离等）。
- 扩散模型 \( p_\theta(R|\tau) \) 生成奖励矩阵 \( R \in \mathbb{R}^{N \times A} \)。
- 使用 \( R \) 作为即时奖励训练每个单位的 DQN，然后以胜率相关指标 \( \rho \) 评估策略。
- 通过精英选择 + 截断扩散变异迭代改进扩散模型。

## 框架流程图（Mermaid）

下图完整映射了星际争霸场景的执行流程，所有节点均使用 SMAC 术语。

```mermaid
flowchart TD
    Start([开始]) --> Init[0. 初始化模块<br/>SMAC环境 / DQN / 扩散模型 / 胜率指标]
    Init --> Q[1. 加载或创建基准 Q<br/>（内置脚本战斗一次，记录胜率）]
    Q --> RandDiff[2. 随机初始化扩散模型]

    RandDiff --> PopLoop{对每个个体 i=1..M}
    PopLoop --> InitPolicy[3. 为每个战斗单位初始化随机DQN]
    InitPolicy --> SimCollect[4.1 执行一次SMAC战斗<br/>收集经验 (obs,action) 与 tau<br/>（tau = 各单位最终观测）]
    SimCollect --> GenReward[4.2 扩散模型生成奖励矩阵 R (N×A)]
    GenReward --> BuildData[4.3 用R替换原始奖励，构建训练数据]
    BuildData --> TrainDQN[4.4 训练每个单位的DQN]
    TrainDQN --> SimEval[4.5 再次战斗，计算胜率指标 ρ]
    SimEval --> StoreInd[存储个体 (tau, R, ρ, 经验库)]
    StoreInd --> PopLoop

    PopLoop --> |M个个体完成| Population[初始种群]

    Population --> EvoLoop{进化轮次 k=1..K}
    EvoLoop --> TrainDiff[5.1 用种群中所有 (tau,R) 训练扩散模型]
    TrainDiff --> SelectElite[5.2 基于ρ的温度采样选择精英]
    SelectElite --> MutateLoop{对每个精英}
    MutateLoop --> TruncMut[5.3.1 截断扩散：<br/>精英奖励加噪→部分去噪，产生变异奖励]
    TruncMut --> Rebuild[5.3.2 用原经验+变异奖励重建训练数据]
    Rebuild --> Retrain1[5.3.3 在原DQN上训练一次]
    Retrain1 --> Resim[5.3.4 新DQN再次战斗，收集新tau和经验]
    Resim --> Regenerate[5.3.5 根据新tau生成新奖励]
    Regenerate --> Retrain2[5.3.6 再次训练DQN]
    Retrain2 --> Reeval[5.3.7 最终战斗，计算新ρ]
    Reeval --> StoreMut[存储变异个体]
    StoreMut --> MutateLoop

    MutateLoop --> |所有精英处理完| Merge[5.4 合并原种群与变异种群]
    Merge --> TopM[5.5 保留ρ最高的M个个体]
    TopM --> Record[5.6 记录本轮最优ρ到CSV]
    Record --> EvoLoop

    EvoLoop --> |K轮结束| End([结束])
## 分步骤说明（星际争霸版）
以下每一步对应 Mermaid 图中的节点，描述其功能和在 SMAC 场景中的具体含义。

### 步骤 0 – 模块初始化
根据 YAML 配置动态加载以下模块（需符合抽象基类）：

环境：SC2Environment – 封装 SMAC，提供 simulate_collect 和 simulate_evaluate。

DQN模块：SC2DqnModule – 管理每个单位的 DQN 网络，用生成的奖励训练。

扩散模型：SC2DiffusionModel – 条件噪声预测网络，输入 tau，输出奖励矩阵。

Q提供者：SC2QProvider – 加载或运行内置脚本得到基准胜率 Q。

指标：SC2Metric – 比较当前策略与 Q 的胜率，计算 ρ。

精英选择器：TemperatureEliteSelector – 按 ρ 进行温度采样。

并行执行器：SerialExecutor 或 RayExecutor。

### 步骤 1 – 读取或创建 Q
调用 SC2QProvider.load_or_create_q(environment)。
若缓存不存在，则使用 全为 None 的策略列表（即 SMAC 内置脚本）运行一次完整战斗，将返回的 battles_won / battles_game 等统计作为基准 Q 并缓存。

### 步骤 2 – 随机初始化扩散模型
SC2DiffusionModel.init_random() 创建条件 UNet 和噪声调度器（DDPM/DDIM），模型参数随机。

### 步骤 3~4 – 构建初始种群（重复 M 次）
每个个体独立执行以下子步骤：

3. 初始化随机策略
SC2DqnModule.init_random_policies(num_agents=N) 为每个战斗单位生成一个随机初始化的 DQN 网络（输入观测维度，输出动作价值）。

4.1 仿真并收集经验库与 tau
SC2Environment.simulate_collect(policies) 运行一次 SMAC 对局：

记录每个时间步的观测 obs 和选择的动作 action，存入 experience_buffers（奖励暂时填 0）。

对局结束后，将最后一个时间步的所有单位观测堆叠成 tau，形状 (N, obs_dim)。

4.2 根据 tau 生成奖励 R
SC2DiffusionModel.generate_reward(tau) 使用 DDIM 采样生成奖励矩阵 R，形状 (N, A)。
R[i,a] 表示单位 i 执行动作 a 时应获得的即时奖励。

4.3 构建 DQN 训练数据
SC2DqnModule.build_training_data(experience_buffers, R) 将每个经验中的奖励占位符替换为 R[i, action]，形成 (state, action, target_Q) 三元组。

4.4 训练每个智能体的 DQN
SC2DqnModule.train_per_agent(training_data) 对每个单位的 DQN 进行 MSE 回归，使 Q(state, action) 逼近生成的 target_Q。

4.5 再次仿真并计算 ρ
SC2Environment.simulate_evaluate(trained_policies) 使用训练后的策略重新战斗，返回 simulation_data（胜率、总奖励等）。

SC2Metric.compute_rho(q, simulation_data) 计算适应度 ρ，例如 ρ = 1/(1+|winrate_q − winrate_cur|)，ρ 越大表示越接近基准性能。

最终将 (tau, R, ρ, experience_buffers, policies=[], metadata) 打包成一个个体。

### 步骤 5 – 进化迭代 K 轮
5.1 训练扩散模型
SC2DiffusionModel.train_on_population(population) 提取种群中所有 (tau, R) 对，训练条件扩散模型（噪声预测 MSE）。

5.2 精英选择
TemperatureEliteSelector.select_elites(population, elite_count) 根据 ρ 值通过 softmax 温度采样选出精英个体。

5.3 对每个精英个体做变异
对每个精英依次执行：

截断扩散变异奖励：generate_reward_truncated(tau, R, add_noise_steps, denoise_steps) 在原奖励基础上先加噪再部分去噪，生成邻域内的新奖励 R_mut。

重建训练数据：用原经验库 + R_mut 调用 build_training_data。

训练一次 DQN：在旧 DQN 上继续训练一个 epoch。

再次仿真：用更新后的 DQN 调用 simulate_collect，得到新经验库和新 tau_new。

再次生成奖励：generate_reward(tau_new) 得到 R_new。

再次训练 DQN：用新经验库和 R_new 再次调用 build_training_data 和 train_per_agent。

最终评估：simulate_evaluate + compute_rho，得到新的 ρ。
将变异后所有信息封装为新个体。

5.4 合并种群
原种群 + 所有变异产生的新个体。

5.5 保留 top‑M 个体
按 ρ 降序排序，只保留前 M 个。

5.6 记录最优 ρ
将当前轮的最优 ρ 追加写入 CSV 文件（配置中指定路径）。
