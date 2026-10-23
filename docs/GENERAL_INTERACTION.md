# 低显存通用交互原型

本文记录第一轮四任务共享残差原型，独立物理检查为 0/4 成功。
后续已改为先训练并认证单个技能，见 [首个滑板技能](FIRST_SKATEBOARD_SKILL.md)。
新方案采用冻结原追踪分支加单个已训练交互分支，不能把本文的共享世界模型学习结构直接套用于新成绩。

2026-10-23：在原生 SuperMimic 第 7000 轮 checkpoint 上接续一个共享交互模型。
当前任务包括原动作追踪、楼梯、搬箱、搬箱越障和滑板。场景与训练闭环已运行；这不等于这些任务已经学会。

## 一个共享模型，保留原动作能力

原 Actor 的 9,116,477 个参数及归一化统计冻结。新建一个 **45,341 参数**的共享交互残差分支，输入包含任务条件、物体状态、目标和地形；四种交互共用此分支，没有每任务一套专家或额外 Teacher/Student。
纯动作追踪 `interaction_mask=0` 通过 `torch.where` 精确选择原 Actor 输出；原追踪场景也完全不添加新碰撞几何。残差初始化为零，交互动作修正限制为归一化动作的 ±0.25。

这保证相同输入下旧动作函数不变，并由原始 checkpoint 的参数/统计逐位比对确认；不代表对改变后的场景或未知输入具有已验证的物理成功率。

新增交互技能采用另外的保护：

1. 五类任务轮流收集，每个训练批次从每类均衡取样，各类独立缓存最多 4096 条经验。
2. 共享交互残差使用上一个已接受模型的回放动作作锚点。
3. 每 50 次更新进行四种任务的固定真实物理评估；任何任务的密集回报低于其历史最佳值的约 2%，就恢复已接受 Actor 并清空其候选优化器状态。
4. `accepted.pt` 是可使用的通过检查版本；`latest.pt` 保留最新世界模型与 Critic，并使用已通过检查的 Actor。原 7000 轮 checkpoint 不被覆盖。

短评估与回放是防退化机制，**不能保证所有新交互技能永久不遗忘**。之后每种技能学成时，需要加入完整任务、不同初始状态和物理参数下的固定回归集，继续保留历史版本。`evaluate_interaction_cpu.py` 单独检查实际任务完成条件，不将密集回报当作成功。

## 物理和学习结构

- 原生 29DoF G1；CPU MuJoCo 做真实采样，CPU 保存参考和 Replay。
- 交互场景加入被动掌部碰撞与固定手部 site；自由箱体为 2 kg，没有焊接、吸附、辅助力。
- 楼梯为四级，级高 8 cm、踏步长 32 cm；搬箱越障场景有 12 cm 障碍。
- 滑板使用 HUSKY 完整自由根、板面倾斜、桥架和轮系。
- World Model 保留原生模型，新增一个共享机器人/物体动力学修正网络；监督来自真实仿真 transition，物体会随预测更新，不是固定参考轨迹。
- Actor 沿原生模型学习路线在 World Model 中做可微想象更新；Critic 接收机器人、物体、上下文和参考 latent。未添加 PPO 循环。
- 想象中的每一步都根据预测位置重新查询地形，并考虑滑板根部与倾斜关节引起的支撑面速度。

当前接口是第一版低资源契约：机器人状态 71 维、物体状态 27 维、上下文 64 维；Actor 原来的 10×99 本体历史、730 参考和 29 动作保留。物体槽表示一个箱体或一套有 7 个内部关节的滑板。
这尚未实现任务书中两个独立对象槽、完整 122/162 维输入或“抱箱滑板”等组合，后续扩展需要版本化数据与 checkpoint 契约。

## 本地数据

保留本地 40 条 LAFAN/G1 CSV；CSV 采样、速度与 goal 已对照原生 MotionLib。
搬箱和搬箱越障均复用本地 VAIC `box_1` 的机器人/物体参考，按关节名称映射，四元数为 wxyz。
`box_stair` 等数据已发现但没有用于当前训练，因为原数据地形与当前障碍几何不同。当前场景也不是 VAIC 场景的等参数复现；箱体初始 z 最低设为 0.181 m 以避免穿地。
楼梯用步行参考，滑板先用原生站姿作平衡课程。参考只提供目标，模型不能直接设置物体的后续运动。
数据、网格、checkpoint 均保留在本地，没有新增下载或重新发布 VAIC 数据。

## 资源配置与第一轮

配置记录在 `configs/g1_general_interaction_low_vram.json`：5 个 CPU 场景、每次收集 4 步、每类 batch 4、想象 horizon 4、前 100 次仅更新动力学/Critic、1000 次总更新。
交互 episode 上限 512 步，训练中回归评估 256 步；这些是初始课程，短于完整任务。
GPU 使用 BF16 MLP、不做编译；训练进程优先级降低，并在迭代间留出短暂空档。

PyTorch 分配器预算低于 4 GiB，另留出 CUDA 上下文空间；每次更新及评估后检查进程显存，超预算报错停止。
这不等于操作系统层面的 GPU 硬隔离。第一轮实测进程显存峰值 **582 MiB**，PyTorch 峰值分配约 **243.5 MiB**。

第一轮 1000 次更新约 344.85 秒完成，五类各收集 4000 条真实 transition。候选更新触发 17 次回退，全部数值有限，训练已结束并释放显存。
本轮最初的程序在结束时对回退后的 Actor 额外做了一次接受检查，因此记录 4 次接受包括最后一次重复检查；后续程序已消除此重复。最终 checkpoint 迭代为 1000，但交互 Actor 的最后一次未回退更新来自第 150 次，WM/Critic 更新到 1000 次。
这说明防退化逻辑在起作用，也说明后续策略学习尚未稳定，不能把完成 1000 次更新解释为任务成功。

## 运行、检查和恢复

在仓库根目录，使用已有 `.venv-local`，明确传入本地路径：

```bash
CUDA_VISIBLE_DEVICES=0 .venv-local/bin/python tools/train_interaction.py \
  --checkpoint /path/to/original_7000/last.pt \
  --motions /path/to/lafan1_supermimic \
  --vaic-root /path/to/VAIC/data/motion/g1/mirobotA \
  --output runs/new_interaction_run --device cuda --vram-mib 4096 \
  --iterations 1000 --eval-steps 256
```

继续旧交互模型时加 `--resume /path/to/accepted.pt`，并提高 `--iterations` 的总目标。必须保留相同原生 foundation 和评估时长；继续前确认参考数据、物理场景和奖励没有发生改变，否则需建立新的对照评估。
模型和优化器状态可恢复，Replay 与 RNG 不恢复，会从五种任务重新采样；不是逐步完全相同的运行续接。不要用新任务覆盖旧任务缓存或只训练最新任务。

独立检查旧能力与新任务：

```bash
CUDA_VISIBLE_DEVICES='' .venv-local/bin/python tools/check_interaction_retention.py \
  --native /path/to/original_7000/last.pt --candidate /path/to/accepted.pt \
  --motions /path/to/lafan1_supermimic --output outputs/retention.json

CUDA_VISIBLE_DEVICES='' .venv-local/bin/python tools/evaluate_interaction_cpu.py \
  --native /path/to/original_7000/last.pt --checkpoint /path/to/accepted.pt \
  --motions /path/to/lafan1_supermimic --vaic-root /path/to/VAIC/data/motion/g1/mirobotA \
  --output outputs/physical.json --steps 512 --save-trajectories
```

第一轮独立检查覆盖 40 条动作、160 个参考采样：原生参数及归一化统计全部相等，追踪动作逐位相同、最大误差 0。还未做新的全长 40 动作物理评测，原先 mjwarp 上的 6000 轮成绩不能当作本次 CPU 交互成绩。

本轮代码通过 60 项 CPU 测试，包括原场景轨迹一致性、真实接触与滑板动力学、参考格式与原生一致性、梯度、共享模型、旧动作保持、均衡回放与退步回滚。HUSKY 资产继续遵循独立 CC BY-NC 4.0 许可。

## 独立任务完成检查

对最终 `accepted.pt` 进行 CPU、最长 512 步（10.24 秒）的真实物理检查，**0/4 达到完成条件**：

| 任务 | 观察结果 |
| --- | --- |
| 楼梯 | 前进约 1.33 m，5.8 s 时触发跌倒判据，未上完四级台阶 |
| 搬箱 | 10.24 s 未倒；双掌同时接触最长 5 帧，没有抬升箱体 |
| 搬箱越障 | 10.24 s 未倒；双掌同时接触最长 4 帧，没有抬箱或越过障碍 |
| 滑板 | 10.24 s 未倒，但双脚同时接触板面比例仅约 22.7%，未达到有效滑行要求 |

这些失败说明下一步需要更有效的接触/抓取目标、分阶段课程和完整任务回归检查。模型没有用焊接或指定物体轨迹伪造成功。训练中 5.12 s 的密集回报门槛不能覆盖这次更长的真实任务检查；不能据此声称新技能已经不会遗忘。
完整记录见 [首轮训练与独立评估报告](../reports/g1_general_interaction_20261023.json)。
