# SMR_deploy

独立的 Unitree G1 29-DoF SuperMimic-Reproduced 部署版本。当前只保留两个动作：

- `neutral_walk_ff_180_R_002__A534`：向前走
- `wave_two_hands_R_001__A533`：挥手

模型输入为 `obs(930)` 和 `goal(680)`，输出为 29 个关节动作。真机使用 Release 的 IR9 `policy.onnx`。

## 训练兼容模型

本部署代码基于 [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)。上游训练代码提供两种 goal 接口：

| 训练模式 | `goal` 维度 | 本项目真机是否兼容 |
|---|---:|---|
| 标准 proprioception-only（论文模式，`--env.anchor-obs` 关闭） | `10 × 68 = 680` | **是，必须使用** |
| BeyondMimic anchor observation（`--env.anchor-obs` 开启） | `10 × 68 + 9 = 689` | 否 |

680 维不是缺少 9 维躯干输出。每个未来帧的 68 维是：关节位置 29、关节速度 29、投影重力 3、机体角速度 3、水平速度 3、根部高度 1。开启 `--env.anchor-obs` 时增加的 9 维是 `torso_link` anchor 的相对位置 3 维和 6D 姿态 6 维，并非额外的关节控制输出。

要训练可由本项目部署的模型，请在上游仓库中使用下面的配置；`future_frames=10`、`history_length=10` 保持默认值，并且**不要添加** `--env.anchor-obs`：

```bash
git clone https://github.com/CuboiLeo/SuperMimic-Reproduced.git
cd SuperMimic-Reproduced
uv sync --group dev
env -u PYTHONPATH uv run sm-train --run-name smr_deploy_compatible \
  --data.npz-dir data/npz/bones_seed \
  --env.num-envs 4096 --policy-trainer.batch-size 4096
```

使用 BONES-SEED 打包库时，也可以使用上游给出的多 GPU 命令：

```bash
env -u PYTHONPATH uv run torchrun --nproc_per_node 4 -m supermimic.scripts.train \
  --run-name smr_deploy_compatible \
  --data.library data/library/bones_seed_compact.pt \
  --env.num-envs 4096 --policy-trainer.batch-size 2048
```

导出前检查训练配置：`env.anchor_obs` 必须为 `false`，`env.future_frames` 必须为 `10`；ONNX 应为 `obs[1,930] + goal[1,680] -> action[1,29]`。上游导出代码会拒绝 `anchor_obs=true` 的 checkpoint，这正是为了避免把 689 维模型误用于没有 anchor 状态输入的真机。

## MuJoCo

安装 `mujoco`、`onnxruntime`、`numpy` 后运行：

```bash
cd SMR_deploy
PYTHONPATH=. mjpython mujoco_e11/mujoco_deploy.py \
  --motion neutral_walk_ff_180_R_002__A534 --align-seconds 8
```

在 MuJoCo 窗口中：`s` 对齐，`a` 播放/重播，`b` 阻尼，`x/y` 切换动作。动作结束后会用策略增益平滑回到首帧待机姿态。

录制连续动作演示：

```bash
PYTHONPATH=. python mujoco_e11/record_demo.py --output videos/mujoco_walk_wave_demo.mp4
```

演示视频：[MuJoCo 连续动作](videos/mujoco_walk_wave_demo.mp4) · [真机部署](videos/real_robot_deployment.mp4)

## 真机

先把机器人吊在吊架上，确认网卡为 `eth0`，并在机器人 PC2 上运行：

```bash
cd ~/SMR_deploy
bash scripts/deploy/bridge/run_e11_keyboard.sh
```

1. 按 `d` 释放 Unitree 高层模式；程序随后等待输入 `yes` 才开始发送底层命令。
2. 按 `s` 对齐首帧后等待 `holding first frame`。此时缓慢放下机器人，让双脚平稳着地并承担主要重量；吊架保持连接，略微保护机器人。
3. 按 `a` 进入策略待机，确认机器人已经站稳后，继续放下吊架直至不再承重，然后按 `x` 播放当前动作。
4. 任意时刻按 `b` 进入 DAMP。吊架和第二个 SSH 终端应始终就绪。
5. 动作结束后程序自动用策略增益平滑回到当前动作首帧：按 `a` 重播当前动作；按 `s` 平滑切到下一个动作，再按 `a`、`x` 播放。

首次真机测试全程保持吊架承重，确认姿态和温度正常后再逐步放松吊架。若 SDK Python 路径不同，设置 `UNITREE_SDK2_PYTHON`；若桥接程序不存在，设置 `UNITREE_SDK2` 后脚本会自动编译。

## 安全

低层命令直接发送到电机。只在空旷区域、吊架保护和可用的 `b` 急停通道下测试；不要把 SSH 断开当作急停。
