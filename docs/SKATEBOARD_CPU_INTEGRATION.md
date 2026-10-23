# 原生 G1 + HUSKY 滑板：CPU 接入

2026-10-23 已将本地 HUSKY 导出场景中的滑板接到 SuperMimic-PERA 的 **29 关节 G1** 上。
这是物理场景、29 维控制接口及接触/物体状态接口的接入，不是 HUSKY 的 23 关节机器人回放，也不是已经训练的滑板 Actor。
原动作追踪任务保持原样，训练继续暂停。没有新增下载、安装依赖或使用 NVIDIA GPU。

## 场景与控制

- `interaction/skateboard_scene.py` 组合原生 `scene_supermimic.xml` 和新增 `assets/objects/husky_skateboard.xml`。
- 保留 G1 全部关节、质量、惯量、位置执行器、增益、力限制、四个球形足底碰撞点/脚及原生控制顺序。
- 滑板保留自由基座、板面倾斜、前后桥架、四轮，共 7 个内部关节；两个桥架耦合等式及原有弹簧均保留。
- 模型总维度 `nq=50, nv=48, nu=36`；动作接口仍为 **29** 维。后 7 个执行器只保留零目标弹簧，轮驱动力为零。
- 归一化动作沿用 `default_q + action_scale * action`，其中 `action_scale = 0.25 * effort_limit / kp`。
- 原生地面、轮子 `condim=6`、摩擦和 board priority 均保留。板面能接收原生 G1 的碰撞位；没有固定机器人到板面的焊接约束，也没有指定板体运动轨迹。
- 为让双脚沿 0.8 m 板长排布，板上 reset 将原生站姿朝向转 90 度，并把足底放在 0.11 m 板面上方 1 mm。它是几何测试初态，不是已训练的平衡初态。

## 状态和接触接口

```python
import numpy as np
from unisupermimic.interaction.skateboard_scene import G1SkateboardCPU

env = G1SkateboardCPU()
state = env.reset("board")  # 或 ground，板会放到远处以独立检查脚地接触
state = env.step(np.zeros(29))  # 四次 0.005 s 物理步 = 50 Hz 控制
```

返回 `robot_qpos[36]`、`robot_qvel[35]` 与独立 `interaction` 字典。后者包含：

- 滑板 `qpos[14]`、`qvel[13]`、7 个内部关节的位置/速度、根部和实际板面的世界位姿及速度。
- 左右脚相对板面的位置、速度；支持面速度考虑根部和倾斜关节产生的 `ω × r`。
- `left_foot_board`、`right_foot_board`、`left_foot_ground`、`right_foot_ground`、`wheel_ground` 五组接触。
- 每组报告接触候选数、有正法向力的接触数、法向合力、作用在脚/轮子上的世界坐标力、接触点和切向相对速度。

自由关节 `qpos` 四元数为 wxyz；原始广义角速度保持 MuJoCo 约定，另有显式世界坐标速度字段。`step` 返回前执行 `mj_forward`，保证状态与接触力对应。直接改 `data` 后也应先执行 `mj_forward` 再读取。

这套字典是当前 CPU 诊断契约。**没有把物体状态悄悄拼进既有训练张量**：现有 WM、Replay、Actor 仍只描述原动作追踪任务；7000 轮 checkpoint 不能直接当作滑板策略。后续需要单独定义交互观测、关节物体动力学预测、奖励和训练任务。

## 完全离线验证

在仓库根目录使用已存在的环境：

```bash
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=disable OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  .venv-local/bin/python -m pytest -q \
  tests/test_skateboard_contacts.py tests/test_skateboard_physics.py tests/test_skateboard_scene.py

.venv-local/bin/python tools/validate_skateboard_cpu.py \
  --output outputs/g1_skateboard_new_check
```

输出目录必须尚不存在。检查入口主动隐藏 CUDA，仅导入 NumPy 和 MuJoCo，不导入 Torch/Warp/UniLab，不创建渲染上下文，不加载 checkpoint。
场景只依赖已在本地的 G1 网格和仓库内的滑板 XML；网格缺失直接报错，不自动获取。7 项纯接触/滑板测试无需 G1 网格；5 项原生 G1 集成测试在没有网格时跳过。

本次结果：**12 passed**，MuJoCo 3.11.0。独立 CPU 检查的 10 项接入断言也全部通过。
可追踪的结果快照见 [CPU 接入报告](../reports/g1_skateboard_cpu_20261023.json)。

| 检查 | 结果 |
| --- | --- |
| 滑板四轮承重 | 合力约 68.18 N，与 6.95 kg 滑板重力一致 |
| 同为 0.5 m/s 初速，运行 0.5 s | 纵向滚动约 0.230 m，横向滑移约 0.0117 m |
| 倾斜/桥架耦合与弹簧 | 复位且等式残差小于 0.001 |
| 原生 G1 在板上，0.4 s | 双脚均有足板支撑，轮地竖直合力约 393.74 N，接近人板总重 395.26 N |
| G1 在地面，板放远处 | 双脚只有足地支撑，滑板独立承重约 68.18 N |
| 接触力方向、旋转支撑面速度 | 两种 geom 顺序、根部和板面转动均通过物理测试 |
| 控制兼容性与资源 | 原 29 执行器参数不变；动作仅控制机器人；不导入 GPU/训练运行时 |

报告在 `outputs/g1_skateboard_integration_20261023/`：完整 XML、JSON 接触快照、0.4 s 姿态、2 s 的高度/接触力 CSV 和 summary。
**两种固定 PD 工况在后续 2 s 内都会跌倒**，报告保留此事实。短时间承重证明接触链路有效，不代表已有持续站板、蹬地或转向策略。

## 来源与许可

滑板来自 `TeleHuman/humanoid_skateboarding` 提交 `d93833e80deff7f927c0b80ef9c435d8b5c488fe` 的 `test_scene/mjlab_scene.xml`，以已验证的导出 demo 版本为准。
板面弹簧为该快照的 `kp=14.67, kd=0.402`，并非训练 Python 配置的另一组参数。原有几何自动推导质量也保留，因此滑板总质量为 6.95 kg。
提取的默认类、板体子树、等式和执行器均经过与本地源文件的逐项对照；source JSON 保存源/提取 SHA256。
该资产和早期 `tools/husky_cpu_demo.py` 适用原项目 **CC BY-NC 4.0**，全文在 `LICENSES/HUSKY-CC-BY-NC-4.0.md`，并在 `NOTICE` 中单独标注。
