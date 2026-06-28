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
