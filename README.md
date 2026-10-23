# SuperMimic-PERA

基于 [Motphys/Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic) 的 G1 SuperMimic-PERA 研究代码。
当前先建立 G1 全身动作追踪基线，再开发真实手物交互、地形、楼梯和移动支撑。
动作追踪基线沿用原生 UniLab / mjwarp、World Model、Actor 和 Critic 训练循环。

**首个交互技能已通过独立验收：自主蹬地滑板。**
2026-10-23 用本地 HUSKY 经验训练新的 29 关节交互分支，1 个标准初态与 8 个保留扰动初态
全部完成 30 秒滑行，前进 39.87–42.13 m。训练显存观测峰值 1028 MiB，零下载，训练现已结束。
原追踪参数/统计、160 个动作采样及 1500 步物理对照保持逐位一致。
详见 [首个滑板技能与复现命令](docs/FIRST_SKATEBOARD_SKILL.md) 和
[实验报告](reports/g1_skateboard_skill_20261023.json)。
当前是冻结原追踪分支加已认证滑板分支，显式选择任务；尚未融合成共享世界模型驱动的通用交互 Actor。

已增加 [低显存通用交互原型](docs/GENERAL_INTERACTION.md)：在冻结的原生 Actor/World Model 上训练
共享交互残差，覆盖追踪、楼梯、搬箱、搬箱越障和滑板场景，CPU 仿真加小规模 GPU 学习。
第一轮 1000 次混合更新已完成，实测进程显存峰值 582 MiB；旧动作参数/统计及 160 个采样输出保持不变。
这轮旧原型独立检查为 0/4 成功，后续改为先学会上述单个滑板任务。

上游基础提交为 `ceafbc863c52b3d613d0e19976794c5a31b9b402`；Git 历史与许可证均保留。
原始上游说明见 [UPSTREAM_README.md](UPSTREAM_README.md) 和
[UPSTREAM_README_zh.md](UPSTREAM_README_zh.md)，其中的演示与公开成绩属于上游。

## 当前状态

2026-10-23 原动作追踪基线按用户要求暂停，日志最后记录约第 **7275** 轮；
完整、可恢复的 `last.pt` 保存于第 **7000** 轮。训练和评估均已停止，不会自动重启。
权重、动作数据、网格、环境和本机运行目录保留在本地，未上传到 Git。

| checkpoint | 完整通过动作 | 归一化回报 | 帧加权回报 |
|---|---:|---:|---:|
| 500 | 0/40 | 0.001653 | 0.001574 |
| 1000 | 20/40 | 0.318787 | 0.334853 |
| 2000 | 27/40 | 0.525477 | 0.551922 |
| 4000 | 28/40 | 0.633161 | 0.662700 |
| 6000 | 34/40 | 0.715678 | 0.729270 |

[实验报告](reports/g1_tracking_baseline_20261023/summary.json)包含暂停状态、评估口径和结果。
基线尚未完成目标 12000 轮；新滑板分支独立认证，持物与楼梯尚未达到成功条件。

已增加 [HUSKY 滑板场景与接触的 CPU 接入](docs/SKATEBOARD_CPU_INTEGRATION.md)：
使用原生 29DoF G1、完整滑板轮系及真实接触，12 项 CPU 测试通过。
早期固定 PD 接触验证尚不具备持续平衡能力；现在另有上述学习后的滑板分支。
运行此检查不会恢复训练、下载资源或使用 CUDA。
滑板资产及早期 HUSKY 演示适配脚本遵循独立的 **CC BY-NC 4.0** 许可，见 NOTICE。

## 本地数据与配置

使用已有 LAFAN1/G1 数据：40 个 CSV、264705 个源帧，每帧 36 列。
G1 为 29DoF，原生资产校验为 `nq=36, nv=35, nu=29`。
数据来源为 `lvhaidong/LAFAN1_Retargeting_Dataset`，其许可独立适用。

[基线配置](configs/g1_supermimic_baseline.json)保留原生网络、10 帧历史/未来参考、
8 步世界模型监督、32 步想象展开、前 500 轮 WM 预热。
为共享 16 GB RTX 5080，环境数与训练批次同比缩为 1024、WM batch 512、Actor batch 1024。
本地使用 PyTorch 2.12.1+cu130、UniLab 1.5.0、MuJoCo/mujoco-warp 3.11.0、Warp 1.16.0。
这不是上游 `uv.lock` 的完全一致安装，也不把本机成绩当作等配置复现。

## 默认离线运行

运行入口默认启用 `UV_OFFLINE`、`HF_HUB_OFFLINE`、`HF_DATASETS_OFFLINE` 和离线日志。
缺少依赖或资产时直接报错，不会自动下载。新增下载须先得到明确许可。

首次配置时，指定已有、兼容的 Python/PyTorch 环境。以下脚本只从完整 uv 缓存打包本地 wheel，
不会下载 PyTorch、CUDA 或其他包。运行脚本的 Python ABI 应与目标环境一致。

```bash
export SUPERMIMIC_PYTHON=/path/to/existing/python
"$SUPERMIMIC_PYTHON" tools/cache_supermimic_wheels.py --dry-run
"$SUPERMIMIC_PYTHON" tools/cache_supermimic_wheels.py
bash tools/run_supermimic.sh setup
bash tools/run_supermimic.sh validate-assets
```

`.venv-local` 继承上述基础环境的 PyTorch/CUDA/Triton，其他依赖放在独立环境中，
不修改基础环境。`SUPERMIMIC_WHEELHOUSE` 可指定已有的离线 wheel 目录。
完整安装仍可参考上游文档，但上游的 `uv sync`/资产下载命令可能使用网络。

## 训练与恢复

仓库上传不会启动训练。以下命令只在用户明确决定继续训练时手动执行：

```bash
# 新训练，指定一个新的输出目录
python3 tools/g1_supermimic_baseline.py start --run-dir /path/to/new/run

# 查看原运行目录的状态
python3 tools/g1_supermimic_baseline.py status --run-dir /path/to/existing/run

# 恢复原运行目录中的完整 last.pt
python3 tools/g1_supermimic_baseline.py start --resume --run-dir /path/to/existing/run
```

启动器仅调度原生训练和评估入口，每 500 轮保存 checkpoint；
第 500、1000 及之后每 2000 轮进行固定评估，覆盖全部 40 个完整动作、从首帧开始、关闭噪声。
恢复读取三网络、归一化和优化器，再重新采集 replay；不声称逐步完全连续恢复。

auto-generated `status.json`、`profile.json`、资产 SHA256 清单、原生 `run_config.json`、
`train_log.csv`、TensorBoard 和 checkpoint 都保存在用户指定的运行目录。

```bash
bash tools/run_supermimic.sh train --help
bash tools/run_supermimic.sh eval --help
bash tools/run_supermimic.sh video --help
bash tools/run_supermimic.sh test
```

原生环境接入验证为 83 项测试通过、2 项跳过；资源检查已覆盖真实采样和完整 32 步
模型展开中的三网络更新。代码上传前另做离线入口和语法检查，不重启已暂停的训练。

## 许可

代码使用 [Apache-2.0](LICENSE)。上游参考实现的 MIT 许可保留在
[LICENSES/MIT-supermimic.txt](LICENSES/MIT-supermimic.txt)，归属说明见 [NOTICE](NOTICE)。
UniLab、UniSim、仿真引擎与数据集适用各自许可。
