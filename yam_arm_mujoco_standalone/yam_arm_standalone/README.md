# YAM 机械臂独立 MuJoCo 模型包

从 agenticSTAR `objectslam-demo` 四格视频管线中抽取的机械臂模型（M7 MuJoCo
仿真所用的那一套），已做成**自包含、开箱即仿**的独立包。`yam/yam_full.xml`
保持 demo 原样未动；为让真实物理 step 能跑起来所需的少量补丁全部放在
`scene.xml` 与脚本里，并有注释说明（见下方"物理启用补丁"）。

## 内容

```
yam_arm_standalone/
├── yam/
│   ├── yam_full.xml    # 机械臂本体 MJCF（demo 原文件，网格相对路径 assets/）
│   ├── scene.xml       # 物理就绪场景：本体 + 地板 + 灯光 + 3 项补丁
│   └── assets/         # 9 个 STL 网格（base/link1-5/gripper/tip_left/right）
├── test_load.py        # 无头自检：加载两份模型、跑物理、验证跟踪与夹爪
├── view.py             # 交互查看器：正弦波驱动 6 关节 + 夹爪开合演示
└── README.md
```

## 环境搭建（任一新机器）

```bash
python -m pip install mujoco   # 演示时用的是 3.13，≥3.0 均可
python test_load.py            # 无头验证，末行应输出 ALL CHECKS PASSED
python view.py                 # 有显示器的环境下打开交互窗口
```

无其他依赖（numpy 随 mujoco 安装）。headless 服务器上只跑 `test_load.py`。
自己写控制脚本时直接 `mujoco.MjModel.from_xml_path('yam/scene.xml')` 即可，
无需任何额外补丁代码。

## 模型速览（I2RT YAM）

| 项目 | 值 |
|---|---|
| 关节 | joint1–6 铰链（6 自由度）+ joint7/8 滑动（双指夹爪） |
| joint1 range | -2.618 … 3.142 rad |
| joint2 range | 0 … 3.665 rad |
| joint3 range | 0 … 3.142 rad |
| joint4 range | -1.693 … 1.571 rad |
| joint5 range | -1.571 … 1.571 rad |
| joint6 range | -2.094 … 2.094 rad |
| joint7/8 行程 | 0 … 0.0469 m（demo 重标定的指尖行程） |
| 执行器 | a1–a8，position 控制，kp = 60/80/60/30/20/15/300/300，无 kv |
| 末端站点 | `grip`（home 位姿下世界坐标约 [0.181, 0, 0.174]） |

## 物理启用补丁（为什么需要、scene.xml 里改了什么）

**背景**：demo 四格视频里的机械臂是**运动学回放**——`mujoco_demo.py` 每帧
直接写 `data.qpos = IK 解` 再 `mj_forward`，物理从未 step 过（demo 注释原话：
kp 15–80 的位置执行器对抗重力会"远远落后于目标"）。因此上游模型按真机标定、
未针对离线物理调优，直接 step 会遇到三个问题。`yam_full.xml` 保持原样，
`scene.xml` 逐一解决：

1. **base↔link1 网格重叠 → joint1 卡死**。qpos0 时安装面处两组网格互相
   穿插，物理接触把 joint1 顶死（实测指令 1.0 rad 只能到 0.004 rad）。
   scene.xml 用 `<contact><exclude body1="base" body2="link1"/></contact>`
   排除该对碰撞。把 `yam_full.xml` include 进你自己的场景时，请照抄这条。
2. **qpos0 指尖互穿 → 夹爪自锁**。两指尖网格在 joint7/8≈0 处深度穿插
   （实测穿插区约 [-0.030, +0.023]），一开仿真接触力就把滑动关节压到
   ≈-0.031，且 ctrl 拉不回来。scene.xml 提供 keyframe `home_open`
   （`qpos = 0 0 0 0 0 0 0.0469 0.0469`），起步时 reset 到它即可；
   闭合空爪时 jaw 停在 ≈0.023（指尖相触），稳定无漂移。
3. **关节零阻尼 → 永不收敛**。上游关节 damping=0、执行器只有 kp 没有
   kv，物理 step 后整条臂在重力下持续摆荡不耗散。scene.xml 在
   `<default><joint damping="1.0"/></default>` 里定义阻尼——**必须写在
   `<include>` 之前**，被包含的元素按文档位置继承主文件当前 default 类
   （已实测生效）；另配 `integrator="implicitfast"` 保证含阻尼积分稳定。

**验收数据**（本机 mujoco 3.13，`test_load.py`）：6 关节指令跟踪最大误差
0.112 rad（纯重力下垂，kp 原样保留）；稳态 1 秒漂移 < 0.0001 rad；空爪闭合
稳定停在 0.023 m。若要精确轨迹跟踪，请自行加大 kp 或做重力补偿——本包不
改动 demo 模型参数。

## 使用要点

- **基座系 = 安装面**：`yam_full.xml` 的 `base` body 位于原点，即机械臂安装
  面坐标系。objectslam-demo 里它被装在 0.72 m 桌面上：
  `pos='0.285 -0.593 0.720' euler='0 0 -2.406'`。要桌面安装时直接改
  `yam_full.xml` 里 `base` body 的 `pos/quat`。
- **scene.xml 的地板在 z=-0.1**：qpos0 实测模型最低几何点约 z≈-0.096（保守
  估计），地板放 -0.1 保证初始姿态零接触、纯背景。
- **控制约定**：8 个 position 执行器直接对应 `data.ctrl[0..7]`；demo 语义里
  夹爪 ctrl 越大越**开**（0 ≈ 全闭，CTRL_BAR=0.012 ≈ 捏住把手环的缝，
  0.0469 = 全开）。
- **两份模型的分工**：`yam_full.xml` 是纯模型（未动一字，适合 include 进
  你自己的场景，记得照抄 base/link1 exclude）；`scene.xml` 是物理就绪的
  完整场景，起步请 reset 到 `home_open` keyframe。

## 来源

- 模型出处：I2RT YAM，github.com/i2rt-robotics/i2rt（MJCF+STL）
- 本包版本：agenticSTAR `objectslam-demo/mujoco/yam/`，相对上游有两处
  demo 侧修改——`grip` site 与指尖行程重标定（0.04695 m，见该仓库
  `scripts/mujoco_demo.py` 注释）。商用前请核对上游仓库许可。
