# AlphaZero 五子棋实验室

[English](README.md) | [简体中文](README.zh-CN.md)

[![CI](https://github.com/LouissMa/AlphaZero-Gomoku-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/LouissMa/AlphaZero-Gomoku-Lab/actions/workflows/ci.yml)
[![PyTorch backend](https://github.com/LouissMa/AlphaZero-Gomoku-Lab/actions/workflows/pytorch.yml/badge.svg)](https://github.com/LouissMa/AlphaZero-Gomoku-Lab/actions/workflows/pytorch.yml)
[![Container](https://github.com/LouissMa/AlphaZero-Gomoku-Lab/actions/workflows/container.yml/badge.svg)](https://github.com/LouissMa/AlphaZero-Gomoku-Lab/actions/workflows/container.yml)
[![Release](https://img.shields.io/badge/release-1.0.0-7c3aed)](CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-0f766e.svg)](LICENSE)

这是一个面向研究与工程实践的现代化 AlphaZero 五子棋项目。它保留原项目可直接运行的
NumPy 推理模型，同时加入现代 PyTorch 网络、可复现实验、并行自对弈、统计评测、
Gumbel AlphaZero 和交互式网页应用。

这个仓库不只是算法演示：训练可以精确恢复，搜索算法在相同预算下比较，评测结果采用
置信区间，关键证据以版本化 JSON 保存，发布前还会自动检查文档、模型与版本是否一致。

## 核心能力

- 可配置棋盘尺寸与连珠规则，以及 AlphaZero 风格神经网络 MCTS。
- 可配置的 PyTorch 残差策略价值网络，支持混合精度与可选编译。
- 固定随机种子的自对弈训练、持久化经验池、指标记录与中断恢复。
- 多个并行自对弈 actor 共用批量神经网络推理服务，并复用搜索树。
- 交替先后手的评测场，提供 Wilson 置信区间、Elo 估计和模型晋升门槛。
- Gumbel-Top-k、Sequential Halving、Completed-Q 目标及与 PUCT 的等预算比较。
- Canvas 网页棋盘、真实模型选择、策略热力图、价值估计、悔棋和完整复盘。
- 6x6 四连珠与 8x8 五连珠的 NumPy 模型，以及终端和 Pygame 对弈界面。
- Python 3.10-3.13 CI、容器构建验证和标签驱动的发布流程。

## 快速开始

需要 Python 3.10 或更高版本。

```bash
python -m pip install -e ".[web]"
python -m alphazero_gomoku doctor
gomoku serve
```

浏览器打开 `http://127.0.0.1:8000`。如需桌面界面：

```bash
python -m pip install -e ".[gui]"
python gui_play.py
```

也可以运行非 root Docker 服务：

```bash
docker build -t alphazero-gomoku-lab:1.0.0 .
docker run --rm -p 8000:8000 alphazero-gomoku-lab:1.0.0
```

## 可复现训练

安装训练依赖并运行小型端到端实验：

```bash
python -m pip install -e ".[train]"
gomoku train --config configs/train_smoke.toml
```

启动完整实验或从快照精确恢复：

```bash
gomoku train --config configs/train_6x6.toml
gomoku train --resume runs/gomoku-6x6-baseline/checkpoints/step_000050
```

快照保存网络、优化器、经验池、随机数状态、配置和版本元数据。更多说明见
[训练指南](docs/TRAINING.md)。

## 研究与评测工作流

```text
类型化实验配置 -> 固定种子自对弈 -> 持久化经验池 -> 优化与快照
              -> 统计评测场 -> 置信度晋升 -> 网页推理
```

常用命令：

```bash
gomoku benchmark --config configs/train_smoke.toml --games 2 --device cpu
gomoku arena --candidate runs/gomoku-smoke/checkpoints/step_000001 \
  --config configs/arena_smoke.toml --output reports/arena_smoke.json
gomoku compare-search \
  --model runs/gomoku-gumbel-smoke/checkpoints/step_000001 \
  --config configs/compare_search_smoke.toml
```

相关文档包括[并行自对弈](docs/SCALABLE_SELF_PLAY.md)、[评测场](docs/EVALUATION_ARENA.md)
和 [Gumbel AlphaZero](docs/GUMBEL_ALPHAZERO.md)。

## 模型与证据透明度

| 资料 | 说明 |
| --- | --- |
| [捆绑模型卡](models/README.md) | 文件清单、SHA-256、用途、来源缺口与限制 |
| [并行自对弈烟雾报告](benchmarks/smoke_cpu.json) | 批量推理、并行 actor、树复用与硬件元数据 |
| [Gumbel 与 PUCT 报告](benchmarks/gumbel_vs_puct_smoke.json) | 相同模拟预算和确定性比较格式 |
| [评测场烟雾报告](reports/arena_smoke.json) | 先后手轮换、基线、置信区间与 Elo 格式 |

这些报告使用很小的工作负载，作用是证明完整代码路径与证据格式可运行，并不代表普遍
棋力，也不能用于跨硬件比较吞吐量。具体边界见[基准说明](benchmarks/README.md)。

## 开发与发布检查

```bash
python -m pip install -e ".[dev,train,web]"
python -m pytest
ruff check alphazero_gomoku/cli.py alphazero_gomoku/release.py \
  alphazero_gomoku/policy_value_net_pytorch.py alphazero_gomoku/training \
  alphazero_gomoku/evaluation alphazero_gomoku/gumbel alphazero_gomoku/web tests
gomoku release-check
```

路线图八个阶段均已实现，详见[项目路线图](docs/ROADMAP.md)。`v1.0.0` 标签只会在最终
PR 合并到 `main` 且 CI 全绿后创建，发布步骤见[发布手册](docs/RELEASING.md)。

## 已知限制

- 捆绑 NumPy 权重早于现代训练管线，缺少精确种子、经验数据、硬件和受控棋力评测。
- 已提交基准是烟雾测试，不是排行榜成绩。
- 训练高水平模型仍然需要较多算力，默认测试不会进行大规模训练。
- 棋盘引擎用于演示自由规则五子棋，并未覆盖所有比赛规则变体。

## 参与项目

提交代码前请阅读[贡献指南](CONTRIBUTING.md)和[行为准则](CODE_OF_CONDUCT.md)。问题与
需求请使用 GitHub 结构化模板；安全漏洞请按照[安全策略](SECURITY.md)私下报告，普通
使用问题见[支持说明](SUPPORT.md)。版本历史记录在[变更日志](CHANGELOG.md)，引用信息
位于 [`CITATION.cff`](CITATION.cff)。

## 项目来源与许可

本项目现代化改造自 Junxiao Song 的教育项目
[AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)。原始 README 保存在
[`docs/ORIGINAL_README.md`](docs/ORIGINAL_README.md)，并继续采用 MIT 许可证。
