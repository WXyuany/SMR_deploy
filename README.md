# SMR_deploy

这是面向 Unitree G1-29DoF 的 SuperMimic-Reproduced 部署实现。它使用 689 维 anchor policy 输入：930 维历史本体感知输入，加上 680 维动作目标和 9 维 torso anchor 相对位姿。控制器、MuJoCo 验证程序和真机程序使用同一套观测构造、动作缩放、起始对齐和键盘状态机。

## 文件

- `reproduced_controller.py`：689 维 ONNX 控制器、起始对齐和状态机。
- `reproduced_mujoco.py`：MuJoCo 被动窗口验证程序。
- `reproduced_deploy_real.py`：G1-29DoF 真机程序，接收 LiDAR/IMU 融合里程计。
- `lidar_odometry.py`：UDP 位姿接收和 LiDAR 到 pelvis 外参变换。
- `assets/reproduced_policy_lafan1_anchor_5080.onnx`：已验证的 LAFAN1 anchor policy。
- `assets/reproduced_policy_bones_seed_anchor_5080.onnx`：BONES-SEED checkpoint 导出的 policy，需要匹配其训练动作数据。
- `assets/reproduced_motions/`：包含 anchor 字段的动作文件。

## 安装和 MuJoCo 验证

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
./run_mujoco.sh --motion walk1_subject1 --align-seconds 8
```

MuJoCo 窗口获得焦点后：`s` 对齐并保持，`a` 通过姿态检查后播放，`b` 阻尼，`x/y` 在阻尼状态下切换动作。

## 真机部署

真机程序要求先收到 `rt/lowstate` 和新鲜的根部位姿；缺少里程计时不会发送控制命令。里程计发送端向 G1 的 UDP `8765` 端口发送逐行 JSON：

```json
{"position":[x,y,z],"quaternion":[w,x,y,z],"frame":"lidar"}
```

`frame` 可以是 `lidar`、`pelvis` 或 `base`。使用 `lidar` 时必须填写已标定的 LiDAR 到 pelvis 外参；如果发送端已经输出 pelvis 位姿，则使用 `pelvis` 并将外参保持为单位变换。

在 G1 上运行：

```bash
python3 reproduced_deploy_real.py \
  --interface eth0 --robot g1-29dof \
  --policy assets/reproduced_policy_lafan1_anchor_5080.onnx \
  --motion walk1_subject1 --align-seconds 8 --odom-port 8765 \
  --lidar-to-pelvis-pos 0 0 0 \
  --lidar-to-pelvis-quat 1 0 0 0
```

程序启动后保持阻尼。确认外部位姿连续后按 `s`，看到 `holding first frame` 后松开再按 `a`。任何时候按 `b` 进入阻尼；`Ctrl+C` 会先发送约 0.5 秒阻尼再退出。

## 导出自己的 checkpoint

```bash
PYTHONPATH=/path/to/SuperMimic-Reproduced/src \
python3 reproduced_policy_export.py \
  --checkpoint ckpt_latest.pt \
  --output assets/my_policy.onnx
```

导出的输入必须是 `(1,930)` 和 `(1,689)`，输出必须是 `(1,29)`。动作文件必须包含 `goal`、`joint_pos`、`joint_vel`、`root_pos`、`root_quat`、`root_ang_vel_b`、`anchor_pos_w`、`anchor_quat_w` 字段。
