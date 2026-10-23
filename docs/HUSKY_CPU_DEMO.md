# HUSKY 滑板 CPU 演示

2026-10-23：先使用本地 HUSKY 已训练策略检查真实滑板场景，为物体交互开发提供可运行的参照。SuperMimic 动作追踪训练继续暂停。本次没有训练新权重，也没有将 HUSKY 权重转换为 SuperMimic-PERA 策略。

## 本地来源与资源

- 源码：`/home/tcl/Desktop/start/humanoid_skateboarding`，上游 `TeleHuman/humanoid_skateboarding`。
- 提交：`d93833e80deff7f927c0b80ef9c435d8b5c488fe`。
- 策略：原项目 `ckpts/test.onnx`；场景：`test_scene/mjlab_scene.xml`。
- 复用 `holomotion_train` 环境内的 MuJoCo 3.5.0、ONNX Runtime 1.23.2、NumPy、Pillow、imageio。
- 推理强制使用 `CPUExecutionProvider`；物理使用 MuJoCo CPU；视频使用 Mesa llvmpipe 软件渲染和 CPU H.264 编码。
- 无新增下载，无安装环境，无启动 GPU 训练。运行中检查 NVIDIA compute process 列表为空。

控制与观测根据 HUSKY `test_scene/sim.py` 改编，保留 23 关节、395 维历史观测、50 Hz 策略和 200 Hz 仿真。原示例的循环次数使用物理步数、但每次执行一个控制步，会延长实际仿真时间；这里按控制频率严格执行 30 秒。滑板由原场景中的自由底座、倾斜板、转向架、轮子和接触动力学驱动，没有额外给定板体运动轨迹。

HUSKY 代码和本适配脚本遵循原项目 **CC BY-NC 4.0**；该适配脚本不因位于本仓库而改变授权。原模型和网格保留在本地原项目目录中。

## 复现

在本仓库根目录执行（输出目录必须尚不存在）：

```bash
/home/tcl/miniconda3/envs/holomotion_train/bin/python tools/husky_cpu_demo.py \
  --husky-root /home/tcl/Desktop/start/humanoid_skateboarding \
  --output outputs/husky_cpu_new_run \
  --seconds 30 --scenario straight --render
```

转向命令测试使用 `--scenario turn`：目标速度保持 1 m/s；第 6–18 秒目标航向为 0.35 rad，其余为 0 rad。省略 `--render` 可只运行仿真并保存轨迹和指标。命令 `h` 是航向，不是角速度。

## 结果与边界

| 场景 | 仿真时长 | 板体路程 | 最低骨盆高度 | 数值警告 |
| --- | --- | --- | --- | --- |
| 固定直行命令 | 30 s | 41.56 m | 0.669 m | 0 |
| 改变航向命令 | 30 s | 41.91 m | 0.670 m | 0 |

两次均未触发骨盆高度低于 0.45 m 的跌倒诊断。它们只是固定初始条件下的两个 rollout，不能据此报告鲁棒成功率或精确速度/航向跟踪达标：实际平均速度大于 1 m/s 的命令。画面使用跟随相机，地面网格提供运动参照。视频时间对应真实仿真时间。

输出目录分别为 `outputs/husky_cpu_straight_20261023` 和 `outputs/husky_cpu_turn_20261023`。每个目录包含视频、初/中/末截图、CSV 指标、轨迹 NPZ、策略及场景 SHA256 和运行版本。输出文件被 Git 忽略。

## 接入 SuperMimic-PERA 前的必要工作

1. 保留 HUSKY 演示作为独立基准；它的 23 关节策略不能直接加载到现有 29 关节 SuperMimic Actor。
2. 将板面、轮系、转向约束和足板/足地接触接入我们的 G1 场景，并验证控制映射。
3. 扩展观测、世界模型和奖励以覆盖板体及轮系状态、支撑切换、蹬地、行驶和转向。不能将有内部关节的滑板简单当成单个自由刚体。
4. 显存可用后，再进行我们自己的物体交互训练和多条件评估；本次 CPU 演示不代表这部分已经完成。
