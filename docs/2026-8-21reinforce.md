# 基于 REINFORCE 的 SUMO 路网权重策略模型训练过程

## 训练流程

1. 从训练数据集中采样一个 SUMO 状态：

   ```text
   s = (OD, tau)
   ```

   其中，`OD` 表示当前交通需求，`tau` 表示当前路网的排队长度状态。

2. 将 `OD` 和 `tau` 输入策略模型：

   ```text
   pi_theta(r | OD, tau)
   ```

   策略模型输出路网权重 `r` 的概率分布。

   如果 `r` 是连续值，可以令策略模型输出高斯分布参数：

   ```text
   mu_theta(OD, tau), sigma_theta(OD, tau)
   ```

   然后定义：

   ```text
   r ~ Normal(mu_theta, sigma_theta)
   ```

   即策略模型不是直接输出唯一的路网权重，而是输出一个分布，并从该分布中采样候选路网权重。

3. 对同一个状态 `(OD, tau)`，从策略分布中采样 `S` 个候选路网权重：

   ```text
   r_1, r_2, ..., r_S
   ```

   每个 `r_s` 都表示一种可能的路网权重方案。

4. 将每个候选方案输入扩散模型进行评价：

   ```text
   R_s = F_psi(OD, tau, r_s), s = 1, 2, ..., S
   ```

   得到对应奖励：

   ```text
   R_1, R_2, ..., R_S
   ```

5. 计算当前 batch 内的平均奖励作为 baseline：

   ```text
   b = 1 / S * sum_s R_s
   ```

   `baseline` 表示当前这些候选方案的平均水平，用来降低 REINFORCE 训练时的方差。

6. 计算每个候选方案的 advantage：

   ```text
   A_s = R_s - b
   ```

   如果 `A_s > 0`，说明该路网权重 `r_s` 优于当前平均水平，训练时应提高策略模型生成该类权重的概率。

   如果 `A_s < 0`，说明该路网权重 `r_s` 低于当前平均水平，训练时应降低策略模型生成该类权重的概率。

7. 计算每个采样动作的对数概率：

   ```text
   log pi_theta(r_s | OD, tau)
   ```

   它表示策略模型在当前输入 `(OD, tau)` 下生成该候选路网权重 `r_s` 的概率大小。

8. 构造 REINFORCE 损失函数：

   ```text
   L(theta) = - 1 / S * sum_s [A_s * log pi_theta(r_s | OD, tau)]
   ```

   该损失函数的作用是：

   ```text
   提高高奖励方案 r_s 的生成概率；
   降低低奖励方案 r_s 的生成概率。
   ```

9. 使用梯度下降更新策略模型参数：

   ```text
   theta <- theta - alpha * grad_theta L(theta)
   ```

   其中，`alpha` 为学习率。

10. 重复上述过程，直到训练轮数达到设定值，或者策略模型生成的路网权重在扩散模型评价下趋于稳定。

## 最终算法

```text
输入：
    训练数据集 D = {(OD, tau)}
    策略模型 pi_theta
    扩散奖励模型 F_psi
    每个状态的采样次数 S
    学习率 alpha
    训练轮数 E

输出：
    训练后的策略模型 pi_theta，即已学习完成的神经网络参数 theta。
    在实际使用时，给定新的 OD 和 tau，pi_theta 可以生成对应的路网权重 r。

for epoch = 1, 2, ..., E do
    从 D 中采样一个 batch：
        {(OD_i, tau_i)}_{i=1}^B

    for 每个状态 (OD_i, tau_i) do
        将 (OD_i, tau_i) 输入策略模型 pi_theta
        得到路网权重 r 的分布

        从 pi_theta(r | OD_i, tau_i) 中采样 S 个候选权重：
            r_i,1, r_i,2, ..., r_i,S

        for s = 1, 2, ..., S do
            用扩散模型评价候选权重：
                R_i,s = F_psi(OD_i, tau_i, r_i,s)

            计算该候选权重的对数概率：
                log pi_theta(r_i,s | OD_i, tau_i)
        end for

        计算 baseline：
            b_i = 1 / S * sum_s R_i,s

        计算 advantage：
            A_i,s = R_i,s - b_i
    end for

    计算 REINFORCE loss：
        L(theta) = - mean_{i,s} [A_i,s * log pi_theta(r_i,s | OD_i, tau_i)]

    更新策略模型参数：
        theta <- theta - alpha * grad_theta L(theta)
end for
```
