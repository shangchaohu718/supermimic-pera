# 第一个已认证交互技能：自主蹬地滑板

2026-10-23，新训练的滑板分支在原生 29 DoF G1 上通过 **9/9 个 30 秒真实物理检查**。
其中 1 个为标准初态，8 个为未参与训练或模型选择的固定扰动初态。
各次前进 39.87–42.13 m，右脚板面有效接触比例最低 99.26%，没有跌倒、非足部借撑或物理警告。
这验证了小幅初态扰动下的直线自主蹬地滑行；不代表速度指令精确跟踪、任意地形或通用交互已经完成。

## 为什么先改变训练方式

之前同时训练楼梯、搬箱、越障、滑板的共享残差，独立长评估 0/4 成功。
其中滑板静止参考与前进奖励冲突，密集回报也没有排除下板站立。
现阶段按用户要求只解决一个任务，使用本地 HUSKY 专家提供真实仿真经验和纠正标签。
全程没有下载、外加推进力、焊接或指定物体运动轨迹。

原生 G1 四球足底直接运行 HUSKY 专家会约 6.5 秒跌倒。
滑板场景可显式选择 `skateboard_foot_contacts="husky_capsules"`，采用 HUSKY 七胶囊足底。
机器人仍为原生 29 关节，质量、惯量、执行器、关节范围与动作缩放完全保持；
原动作追踪继续加载原始 XML，其他任务默认碰撞不变。

## 学到了什么，使用什么模型

- 原 SuperMimic 第 7000 轮 Actor：9,116,477 参数，参数及归一化冻结。
- 新滑板分支：370,717 参数，MLP 395→512→256→128→29，从随机权重开始训练，没有复制专家网络权重。
- 输入为 5 帧 HUSKY 约定的本体/命令/历史动作/相位观测，输出原生 29 维归一化关节动作。
- 先行为克隆，再在学生实际访问的状态上用本地专家纠正标签做 DAgger；共 8 轮、13,500 次梯度更新、76,216 条本地经验。
- 后期采集完全执行学生动作；独立验收只调用保存的 PyTorch 学生，不导入 ONNX 专家或训练器。

`RetainedSkillPolicy` 是单个可加载 checkpoint，显式选择 `tracking` 或 `skateboard`。
它目前采用分支保留能力，**不是已经融合多个任务的共享通用 Actor**；本轮不更新 World Model/Critic。
旧共享世界模型原型保留为研究路线，但其成绩与本轮模仿/DAgger 成绩分别记录。
跨任务自动切换、搬箱、楼梯以及抱箱滑板仍待训练和独立验收。

## 资源与结果

初次采样到最后开发评估约 140.1 秒。CPU 执行 MuJoCo、专家标签、学生 rollout 和全部独立验证；
GPU 只做小 MLP 更新，训练进程观测显存峰值 **1028 MiB（约 1.01 GiB）**。
Torch 分配器上限 2048 MiB，每 100 次更新检查进程显存，超过 3800 MiB 停止；这不是操作系统硬隔离。
训练已结束，GPU 显存已释放。渲染使用 Mesa llvmpipe 与 CPU libx264。

独立标准要求：连续 30 秒，前进至少 5 m，骨盆相对板高度至少 0.45 m，
姿态向上分量至少 0.65，右脚有效板面支撑比例及尾部 5 秒均至少 80%，
无板面接触间隔不超过 0.3 秒，人板水平距离不超过 0.65 m，
禁止膝、手、身体借撑，右脚不可持续踩地，左脚允许正常蹬地。
全部固定初态与阈值在训练结果选择之前确定，详见 `tools/skateboard_skill_metrics.py`。

原能力检查独立读取原始第 7000 轮文件：26 个参数/缓冲区完全一致，
40 条动作的 160 个真实参考采样输出逐位一致，最大误差 0；
3 条动作各 500 步，原始场景与新模型追踪分支的动作、qpos、qvel 每步逐位相同。
这是保持等价的证据，不把它解释为基线 40 个动作全部成功。

## 防止后续遗忘

认证文件为本地运行目录中的 `certified_skateboard_v1.pt`，原始第 7000 轮文件仍单独保存。
认证工具要求同一个候选 SHA 对应的全量物理验收和原能力检查都通过，且拒绝覆盖已有认证版本。
加载认证文件时，滑板参数与归一化摘要必须匹配，恢复参数冻结和统计冻结；调用 `train()` 也不会解除冻结。
后续新增任务必须保留原追踪与这个滑板模块，并再次通过保存的 9 个物理场景。
若未来要合并或微调这些模块，需建立新的候选版本，通过旧能力完整回归后才替换，不能直接覆盖认证版本。
这保护当前显式分支；未知环境的普遍不退化仍需要更广的验证。

## 本地复现

在仓库根目录使用已有 `.venv-local`，以下命令不安装或下载任何文件。
本轮配置在 `configs/g1_skateboard_skill_local.json`；完整汇总在 `reports/g1_skateboard_skill_20261023.json`。

```bash
# 新训练必须使用新的输出目录，避免覆盖已认证技能。
.venv-local/bin/python tools/train_skateboard_skill.py \
  --native /path/to/original_7000/last.pt \
  --expert /path/to/local/humanoid_skateboarding/ckpts/test.onnx \
  --output runs/new_skateboard_run --device cuda:0

# 部署/验收不需要专家文件。
CUDA_VISIBLE_DEVICES='' .venv-local/bin/python tools/evaluate_skateboard_skill.py \
  --checkpoint runs/new_skateboard_run/candidate.pt \
  --output outputs/new_skateboard_physical.json --save-trajectories

CUDA_VISIBLE_DEVICES='' .venv-local/bin/python tools/check_retained_skill_policy.py \
  --native /path/to/original_7000/last.pt \
  --candidate runs/new_skateboard_run/candidate.pt \
  --motions src/unisupermimic/assets/motions/g1/lafan1_supermimic \
  --output outputs/new_skateboard_retention.json

CUDA_VISIBLE_DEVICES='' .venv-local/bin/python tools/certify_skateboard_skill.py \
  --candidate runs/new_skateboard_run/candidate.pt \
  --physical outputs/new_skateboard_physical.json \
  --retention outputs/new_skateboard_retention.json \
  --output runs/new_skateboard_run/certified_skateboard_v1.pt
```

本轮视频和完整轨迹在 `outputs/skateboard_student_certification_20261023*`。
模型、经验、视频和本地数据不加入 Git。HUSKY 派生碰撞/观测适配及专家来源遵循
CC BY-NC 4.0，见 NOTICE 和 `LICENSES/HUSKY-CC-BY-NC-4.0.md`。
