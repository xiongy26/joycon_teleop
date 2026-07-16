# 把 GUI 实时环路 IK 改为准解析解

## 目标与范围

将 `cartesian_control_gui.py` 实时遥操作环路中的 `mink.solve_ik`(速度级 QP 数值 IK)替换为**准解析 IK**:位置部分用 L2/L3/L4 三平行轴的封闭解,姿态部分用 1 步雅可比修正。保留 mink 作为奇异/失败时的回退。

**只改 GUI 实时环路**(`_solve_velocity_ik`)。SDK 的 `kinematics.py` 数值 IK 本次不动。

## 几何依据(已用 MuJoCo 验证)

零位世界系下各关节轴:

| 关节 | 锚点 | 轴方向 | 说明 |
|---|---|---|---|
| L1 | (0, 0, 0.06) | (0,0,1) | 腰,竖直 |
| L2 | (-0.0096, 0, 0.129) | (0,-1,0) | 肩,水平 |
| L3 | (0.1804, 0, 0.129) | (0,-1,0) | 肘,与 L2 平行 |
| L4 | (-0.0046, 0, 0.193) | (0,-1,0) | 与 L2/L3 平行 |
| L5 | (-0.0538, 0, 0.231) | (0,0,-1) | 腕 |
| L6 | (-0.047, 0, 0.192) | (1,0,0) | 腕 |

- **L2/L3/L4 三轴结构性平行**(点积恒=1.0)→ 满足 Pieper 条件,位置可解析
- L5⊥L6 且相交 → 腕心 = L5∩L6 = (-0.0538, 0, 0.192)
- L4⊥L5 但**有 4.92 cm 偏置,不相交** → 非标准球形腕,姿态与位置耦合
- 末端 site 到腕心偏置 ≈ 0.187 m(沿 L6 轴),腕心到末端在 L6 坐标系下为常量
- L6 限位 [-0.79, 0.79](注意比 URDF 的 ±1.57 窄,以 mujoco `jnt_range` 为准)

## 实现方案

### 1. 新建 `core/analytic_ik.py`

独立模块,复用 `ik_core.py` 已加载的 mujoco `model`/`data`、`arm_qpos_indices`、`ARM_JOINT_LIMITS_*`。不重复加载模型。

**几何参数提取**(模块加载时从 mujoco 零位一次性算出,缓存为常量):
- 腰高 d1、肩偏置、大臂长 a2(L2→L3 = 0.19)、小臂长 a3(L3→L4 投影)、L4→腕心偏置
- 末端到腕心向量 `r_ee_wrist`(在 L6 坐标系下,常量)
- L2/L3/L4 公共轴方向、L1 轴方向

**核心函数**:
```python
def solve(target_pos, target_rot, q_ref, *, pos_tol=1e-4) -> tuple[np.ndarray, float] | None:
    """准解析 IK。返回 (q, err_norm) 或 None(不可达/奇异)。"""
```

求解步骤:
1. **腕心目标**:`p_w = target_pos - R_target @ r_ee_wrist_world`(把末端姿态的偏置扣掉)
2. **位置解析(L1,L2,L3,L4)**:
   - θ1(腰):由 `p_w` 的 x/y 投影 + 肩偏置反解(atan2,2 解:前/后腰)
   - 在 θ1 决定的平面内,L2//L3//L4 三平行轴 → 腕心平面坐标方程化为关于 θ3 的多项式(目标用 `numpy.roots` 求实根)
   - θ4 由腕心偏置约束回代;θ2 由几何回代
   - 产出至多 4 组位置解(肘上/肘下 × 腰前/后)
3. **姿态修正(θ4,θ5,θ6,1 步雅可比)**:
   - 对每组位置解,固定 θ1/θ2/θ3,用 L4/L5/L6 的 3×3 雅可比块一步解姿态残差(DLS,阻尼防腕奇异)
   - 若 L4 已由位置确定,则姿态只解 θ5/θ6 + θ4 微调
4. **多解择近**:计算各组 `‖q - q_ref‖`,取最近的一组(保证实时连续,避免跳变)
5. **限位夹紧**到 `ARM_JOINT_LIMITS_*`,任一关节越界则该解判废

**FK 对拍自检函数**:
```python
def self_test(n=1000):  # 随机 q -> FK -> IK -> 比对
```
位置误差 < 1e-4 m、姿态误差 < 1e-4 rad 才算通过。模块 import 时不自动跑,提供命令行入口 `python -m core.analytic_ik`。

### 2. 改 `cartesian_control_gui.py`

**`__init__`/配置**:
- 新增 `self._use_analytic_ik` 开关,读 `config.json` 的 `ik.use_analytic`(默认 `true`);config 不存在时默认开
- 保留 `_init_mink`(mink 作回退,且 `mink.Configuration` 仍用于 FK 读取/限位检查)

**新增 `_solve_analytic_ik(target_pos, target_rot, n_steps, caller)`**,签名与 `_solve_velocity_ik` 一致,内部:
- 起点限位检查(复用 `_arm_limit_violation_reason`)
- 一次性 `q_target = analytic_ik.solve(target_pos, target_rot, q_current)`
  - 失败/None → 回退调用 `_solve_velocity_ik`(记日志)
- **平滑**:`dq = q_target - q_current`,按 `‖dq‖ > max_step` 缩放(沿用 `ik_core` 限速思路),`max_step` 由 `n_steps` 反推(替代原"迭代次数"语义为"插值步数")
- 关节限位夹紧 + 越界回滚(复用现有逻辑)
- 更新 `self._configuration` 与全局 `data`
- 收敛检查 + `_log_pose`(复用)

**上层调用点**(gamepad `:2730`、pos `:2878`、rot `:2943`):不变,仍调 `_solve_velocity_ik`。在 `_solve_velocity_ik` 内首行分流:
```python
if self._use_analytic_ik:
    return self._solve_analytic_ik(target_pos, target_rot, n_steps, caller)
# else 原 mink 逻辑
```
这样改动最小、可单开关回退。

### 3. 验证

1. **离线对拍**:`python -m core.analytic_ik` 跑 1000 随机位姿 FK↔IK,误差达标
2. **仿真回归**:仿真模式下手柄遥操作,确认末端无跳变、不越限、奇异姿态自动回退 mink
3. **实机回归**:小幅度位姿命令,确认关节连续平滑(对比改动前日志的 `pos_err`/`steps`)

## 风险与处理

| 风险 | 处理 |
|---|---|
| 三平行轴多项式推导错 | SymPy 辅助推导 + FK 对拍 1000 例强制通过 |
| 多解切换导致末端跳变 | 择近 `q_ref` + 单步 `max_step` 限速 |
| 腕奇异(L5≈0)姿态雅可比退化 | DLS 阻尼;残差过大时回退 mink |
| 丢掉 mink QP 的平滑/限位 | 在 `_solve_analytic_ik` 内补限速 + 夹紧 + 越界回滚(见上) |
| 目标超出工作空间 | `solve` 返回 None → 回退 mink(mink 也会失败则上报) |

## 不做

- 不改 SDK `kinematics.py` / `interface.py` 的数值 IK(用户本次只选 GUI 路径)
- 不删除 mink 依赖(作回退)
- 不改运动学模型/URDF
