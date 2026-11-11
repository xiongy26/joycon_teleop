# EL-A3 Cartesian Control GUI

基于 PySide6 + MuJoCo + mink 的 EL-A3 机械臂末端笛卡尔空间控制程序。

## 功能

- GUI 按钮控制末端沿 X/Y/Z 轴平移，Roll/Pitch/Yaw 旋转，可配置步长
- Joy-Con (L) 手柄遥操作：摇杆 XY 平移 + 按键姿态控制（默认）+ IMU 陀螺仪姿态控制（可选）
- Xbox 等通用手柄支持（通过 `/dev/input/jsN`）
- 仿真模式：MuJoCo 物理仿真 + 3D 可视化窗口
- 真机模式：通过 CAN 总线连接实体 EL-A3 机械臂，MuJoCo 窗口同步显示
- 实时显示末端位姿，详细控制日志记录

## 依赖

```bash
pip install PySide6 mujoco mink qpsolvers daqp numpy
```

Joy-Con 手柄控制额外需要：
```bash
pip install python-evdev
```

真机模式额外需要安装 el_a3_sdk：
```bash
cd el_a3_sdk && pip install -e .
```

独立部署 `joycon_teleop` 时，还需要随包携带从项目根目录复制来的
`ela3_motion/` 部署层。GUI 的真机适配器通过
`ela3_motion.ELA3RealtimeController` 连接、读取反馈并提交 50 Hz 实时关节目标，
不再直接维护本地真机轨迹队列逻辑。

## Joy-Con IMU 权限设置

Joy-Con 的摇杆/按键设备和 IMU 传感器是两个独立的 input 设备，普通用户默认只能访问前者。要使用 IMU 姿态控制，需要将当前用户加入 `input` 组：

```bash
sudo usermod -aG input $USER
```

执行后需要**注销桌面重新登录**（或重启电脑）使组权限生效，因为 Linux 的组权限在登录时加载，当前会话不会自动刷新。可通过以下命令验证 IMU 设备是否可访问：

```bash
# 查看 Joy-Con IMU 对应的 event 设备号
cat /proc/bus/input/devices | grep -A 4 "Joy-Con.*IMU"
# 检查权限（应显示当前用户有 rw 权限）
ls -la /dev/input/eventXX   # 替换为上一步查到的设备号
```

> 临时方案（无需重新登录）：`sudo chmod 666 /dev/input/eventXX`

## 运行

```bash
cd joycon_teleop
python run.py
```

## 真机使用

真机模式依赖随 `joycon_teleop` 一起部署的 `ela3_motion` 实时控制层；该层负责
CAN/SDK 连接、安全限位、短轨迹加密和队列提交，GUI 只发送实时关节目标。

1. 设置 CAN 接口：
```bash
sudo ./el_a3_sdk/scripts/setup_can.sh can0 1000000
```

2. 在 GUI 中选择 "Real" 模式，配置 CAN 接口和 Kp/Kd 参数
3. 点击 "Start Viewer" 连接真机

## ACT 数据采集

项目提供了标准 ACT/ALOHA 风格的 HDF5 episode 采集工具，可在零力矩拖动示教时同步保存相机图像和关节状态。

额外依赖：

```bash
pip install h5py opencv-python
```

采集一个 episode：

```bash
python scripts/collect_act_episode.py \
  --task-name pick_cube \
  --camera cam_high=0:640x480 \
  --camera cam_wrist=2:640x480 \
  --rate 30 \
  --output-dir data/act
```

默认流程：

1. 打开相机并连接真机
2. 使能机械臂并进入零力矩重力补偿示教模式
3. 按 Enter 开始录制
4. 拖动机械臂完成示教
5. 按 Ctrl+C 结束录制并保存 HDF5 文件

如果需要固定录制时长：

```bash
python scripts/collect_act_episode.py \
  --task-name pick_cube \
  --camera cam_high=0:640x480 \
  --duration 10
```

检查采集文件：

```bash
python scripts/inspect_act_dataset.py data/act/episode_pick_cube_000000.hdf5
```

HDF5 数据结构：

```text
observations/qpos              float32 [T, 7]
observations/qvel              float32 [T, 7]
observations/images/<camera>   uint8   [T, H, W, 3] RGB
timestamps                     float64 [T]
action                         float32 [T, 7]
```

当前 `action[t]` 定义为下一帧 `qpos` 目标，最后一帧 action 复制最后一帧 `qpos`。这适合先打通 ACT 的 chunked action 训练流程；后续如果改为记录遥操作命令或夹爪离散动作，可以在 `act_data/episode_writer.py` 中替换 action 生成逻辑。

训练一个 ACT-style 策略：

```bash
python scripts/train_act_policy.py \
  --dataset data/act \
  --cameras cam_high,cam_wrist \
  --chunk-size 50 \
  --batch-size 8 \
  --epochs 100 \
  --device cuda \
  --output-dir checkpoints/pick_cube_act
```

如果机器没有 CUDA，使用 `--device cpu`。训练脚本会保存：

```text
checkpoints/pick_cube_act/train_config.json
checkpoints/pick_cube_act/policy_best.pt
checkpoints/pick_cube_act/policy_epoch_*.pt
```

部署训练好的策略到真机：

```bash
python scripts/run_act_policy.py \
  --checkpoint checkpoints/pick_cube_act/policy_best.pt \
  --camera cam_high=0:640x480 \
  --camera cam_wrist=2:640x480 \
  --rate 10 \
  --query-interval 10 \
  --max-target-delta 0.08 \
  --max-joint-velocity 0.6
```

部署脚本会实时读取相机和关节状态，策略输出 action chunk 后逐步发送关节目标。每个关节目标会先按 `--max-target-delta` 限制单步变化，再裁剪到默认关节限位内。默认每步阻塞执行，避免异步命令堆积；只有明确需要时才使用 `--nonblocking`。建议首次上真机时降低 `--rate`、`--max-target-delta` 和 `--max-joint-velocity`，并随时准备急停。

仅验证模型加载、相机读取和动作生成流程时，可以加 `--dry-run`，但脚本仍需要连接机械臂读取当前关节反馈。

部署前安全检查：

```bash
python scripts/preflight_act_policy.py \
  --checkpoint checkpoints/pick_cube_act/policy_best.pt \
  --camera cam_high=0:640x480 \
  --camera cam_wrist=2:640x480 \
  --rate 10 \
  --max-target-delta 0.08 \
  --max-joint-velocity 0.6
```

这个脚本只做检查，不执行策略动作。它会验证 checkpoint、相机帧、关节反馈、单次策略推理，以及首个动作是否能通过实时控制层的 dry-run 安全约束。只想在开发机上检查 checkpoint 结构时，可以用 `--skip-cameras --skip-robot`。

## ACT 仿真验证（MuJoCo，自动采集 + 自动训练）

`act_sim/` 提供 EL-A3 的 MuJoCo 仿真任务（桌面 + 红色方块 + cam_high/cam_wrist 两个任务相机），用于在上真机前验证整条 ACT 管线：脚本化 IK 专家自动生成演示数据（无需人工示教），复用与真机完全相同的 HDF5 格式、训练脚本，然后在 MuJoCo 里做闭环 rollout 评估，输出成功率报告。

一键运行（采集 → 数据检查 → 训练 → 闭环评估 → 报告）：

```bash
python scripts/run_act_sim_pipeline.py                       # 默认 24 条演示、80 epochs、12 次评估
python scripts/run_act_sim_pipeline.py --quick               # 冒烟配置（3 条演示、3 epochs）
python scripts/run_act_sim_pipeline.py --skip-collect --skip-train   # 只评估已有 checkpoint
```

产物在 `runs/act_sim_<时间戳>/` 下：`data/`（HDF5 演示）、`ckpt/policy_best.pt`、`eval/eval_results.json`、`report.md`（成功率表格）与 `eval/rollout_*.mp4`（俯视+腕部并排回放视频）。

仿真评估会沿用真机的关节限位，并默认把每个控制周期的目标变化限制为 `0.08 rad`（可用 `--max-target-delta` 调整）。成功要求最后一步距离小于 3.5 cm，且最后 20% 的轨迹至少一半时间保持在 3.5 cm 内，避免“只碰到一下”被算成成功。

单独使用各模块：

```bash
python -m act_sim.collect --episodes 24 --output-dir data/act_sim   # 只自动采集
python -m act_sim.evaluate runs/act_sim_xxx/ckpt/policy_best.pt --output-dir runs/act_sim_xxx/eval  # 只评估
```

### 浏览器查看器（MuJoCo WebAssembly）

仿真结果可以在浏览器里直接看：查看器用官方 `mujoco-js`（MuJoCo 编译为 WebAssembly）+ three.js 渲染，无需任何插件。

```bash
python scripts/serve_act_sim_viewer.py        # 自动导出最新 run 数据并启动本地服务
# 打开 http://127.0.0.1:8010/web/viewer.html
```

查看器支持三种模式：**交互仿真**（暂停/归位、拖滑块控制 7 个关节指令、移动方块、切换自由/cam_high/cam_wrist 视角、显示接触点）、**演示回放**（回放自动采集的 HDF5 演示，可拖时间轴）、**策略回放**（回放训练后策略的闭环评估轨迹，显示每步 tip-cube 距离）。相机画面与训练时离屏渲染使用同一模型定义。

## Joy-Con (L) 按键说明

```
         ╭────────────╮
         │   [ZL]     │  ← Z 轴上升
         │   [L]      │  ← Z 轴下降
         │  [SL][SR]  │  ← SL Roll 右 / SR Roll 左
         ╰────────────╯

      [↑] 回零          [-]
    [←] [→]            切换 IMU 模式
      [↓] 切换速度

      [摇杆]           [Capture]
    ↕ 前后 → X 轴      未使用
    ↔ 左右 → Y 轴
```

### 摇杆

| 方向 | evdev 轴 | 控制功能 |
|------|----------|----------|
| 左/右 | ABS_X (axis 0) | Y 轴平移 |
| 前/后 | ABS_Y (axis 1) | X 轴平移 |

### 按键（默认姿态控制方式）

| 物理按键 | evdev 编码 | 控制功能 |
|----------|-----------|----------|
| ↓ (方向键下) | BTN_Z (309) / ABS_HAT0Y | 切换速度档位（5 档）+ Pitch 下 |
| ↑ (方向键上) | BTN_DPAD_UP (544) / ABS_HAT0Y | 回零（归位）+ Pitch 上 |
| ← → (方向键) | ABS_HAT0X | Yaw 左/右 |
| **L (肩键)** | **BTN_TL (310)** | **Z 轴下降** |
| **SL (侧面)** | **BTN_TR (311)** | **Roll 右（横滚）** |
| **ZL (扳机)** | **BTN_TL2 (312)** | **Z 轴上升** |
| **SR (侧面)** | **BTN_TR2 (313)** | **Roll 左（横滚）** |
| **- (减号)** | **BTN_SELECT (314)** | **切换 IMU 姿态模式** |

### IMU 姿态控制（可选模式）

默认使用按键控制姿态（SL/SR → Roll，D-pad → Pitch/Yaw）。按 `-` 键可切换到 IMU 模式，此时陀螺仪数据覆盖按键输入，按住手柄转动即可控制姿态。

| IMU 数据 | evdev 轴 | 控制功能 |
|----------|----------|----------|
| 陀螺仪 X | ABS_RX (axis 3) | Roll（横滚） |
| 陀螺仪 Y | ABS_RY (axis 4) | Pitch（俯仰） |
| 陀螺仪 Z | ABS_RZ (axis 5) | Yaw（偏航） |

### 使用流程

**默认模式（按键控制姿态）**：
1. 点击 "Start Viewer" 启动仿真/连接真机
2. 摇杆控制 XY 平移，SL/SR 控制 Roll，D-pad 控制 Pitch/Yaw
3. L/ZL 控制 Z 轴升降，A 切换速度，B 回零

**IMU 模式（可选）**：
1. 点击 GUI 中的"校准 IMU"按钮（手柄需静止放置），等待 1 秒
2. 按 `-` 键切换到 IMU 模式（或勾选"IMU 姿态控制"复选框）
3. 转动手柄控制末端姿态，摇杆仍控制 XY 位置
4. 再次按 `-` 键关闭 IMU 模式，回到按键控制

> **注意**：使用 IMU 模式前必须校准，否则姿态控制会剧烈漂移。每次连接手柄后建议重新校准。

## 贡献者

- **xiongy26** - 项目发起人 & 主要开发者
- **Claude Code** - AI 辅助开发
