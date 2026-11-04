# 中文手语孤立词识别 · Transformer

输入一段边界已确定的手语动作，输出候选词和置信度。本项目用双手关键点比较了两种日常20词分类器的初始化方式，并提供自定义200词推理和双模式摄像头演示。

## 日常20词实验

我负责整理日常词表与手语者划分、构建逐帧特征管线、实现 Transformer 迁移训练和随机初始化对照，并把已有识别模型接入摄像头演示。词表从 NationalCSL-DP 中选择，要求每个词有唯一精确匹配，且十位手语者都有正面样本。手语者和词表在对照实验前固定。[固定词表](realtime_mvp_v2/class_order.csv) · [评测协议](realtime_mvp_v2/evaluation_protocol_draft.json)

| 初始化 | 最佳 epoch | 验证正确数 | 验证 Accuracy | 验证 Macro-F1 |
|---|---:|---:|---:|---:|
| 自定义200词预训练 | 29 | 11 / 20 | **55%** | 0.4583 |
| 随机初始化 | 25 | 4 / 20 | **20%** | 0.1278 |

在相同的 `seed=42`、训练样本和两阶段日程下，迁移模型多识别对了 7 个词。验证集只有一位手语者，每个词仅有一个样本，一条预测就占 5 个百分点。这是一次固定日程下的初始化对照。**日常20词阶段测试尚未评估**。两行指标均来自已有的本地实验。[逐词对照](results/daily20_validation.csv) · [原实验汇总](results/initialization_comparison.csv)

用 Python 标准库即可从逐词预测重新计算上表的 Accuracy 和 Macro-F1。

```bash
python results/check_validation.py
```

### 训练方法

- **数据**　P01–P07 的 140 段用于训练，P08 的 20 段用于验证，P09–P10 的 40 段保留作阶段评测。这两位手语者曾出现在早期项目的错误分析中。
- **特征**　MediaPipe Tasks 在原始画面上提取左右手关键点，保留无手帧，按整个片段均匀采样 30 步。每步有 126 维局部手形、6 维腕部位置、132 维相邻时间步之差，以及两个在场标记。出现或消失的手不参与跨帧差分。
- **模型**　两层 Transformer，隐藏维度 64，4 个注意力头。迁移时加载 31 个兼容张量，重新初始化20类输出层。标准化只用 140 个训练样本拟合，验证集和随机初始化对照复用同一变换。
- **对照**　两组均使用 AdamW，前 10 轮只训练分类头，学习率 `1e-3`。第 11 轮起解冻全模型，学习率 `1e-4`，最多再训 100 轮，早停耐心为 20。最佳轮次依次按验证准确率、Macro-F1、较低的验证损失决定。[实际配置快照](realtime_mvp_v2/daily20_training_config.json)

## 安装

命令从仓库根目录执行。使用 Python 3.10 环境，安装与设备匹配的 PyTorch 后安装依赖。摄像头演示还需要 Tk、图形桌面和可用的摄像头。[PyTorch 安装说明](https://pytorch.org/get-started/locally/)

```bash
python -m venv .venv
# Linux / macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python realtime_mvp_v2/train_daily20_transfer.py --help
python nationalcsl200_stage_v1/infer_nationalcsl200.py --help
python domain_mvp_v1/realtime_sign_demo.py --help
```

### 自备文件

| 使用场景 | 所需文件及默认位置 |
|---|---|
| 提取关键点，或运行摄像头识别 | 官方 Hand Landmarker 模型放入 `models/hand_landmarker.task` |
| 准备日常20词数据 | 按数据源的获取条件准备 `local_data/NationalCSL-DP/Pics/Participant_01.zip` 至 `Participant_10.zip`；另保留 `local_data/NationalCSL-DP/gloss.csv` 供200词准备流程使用 |
| 200词推理、默认摄像头识别、日常20词迁移初始化 | 与仓库200词顺序一致的可信检查点 `nationalcsl200_stage_v1/checkpoints_v2/nationalcsl200_transformer_optimized.pt` |
| 摄像头切换至医疗11词 | 另备旧领域20词检查点 `domain_mvp_v1/checkpoints/domain_mvp20_transfer.pt`，类别顺序见 [领域词表](domain_mvp_v1/class_order.csv) |

模型检查点需包含训练配置、类别顺序、权重和已拟合的标准化参数。仓库发布源码、标签映射和汇总结果。视频、特征、数据压缩包和模型权重由使用者自行准备。

## 运行

**训练日常20词对照。** 特征已按原工程准备好的用户可从第二条命令开始。训练器校验包含 200 项的本地清单和特征路径，训练与选模只加载 train/val 数组。第一条命令会从授权的正面帧序列生成 `realtime_mvp_v2/data/manifest.csv`、原始特征和准备摘要。

```bash
python realtime_mvp_v2/prepare_realtime_mvp_v2.py --dataset-root local_data/NationalCSL-DP
python realtime_mvp_v2/train_daily20_transfer.py --device cpu
python realtime_mvp_v2/train_daily20_scratch.py --device cpu
```

随机初始化对照读取上一步的 checkpoint 和训练报告，使用其中的标准化参数、结构及验证预测作比较，模型参数重新随机初始化。两次运行各自写入 `realtime_mvp_v2/checkpoints/` 和 `realtime_mvp_v2/reports/`。原始记录的精简副本在 `results/`，与新运行输出分开存放。

**自行获取200词初始化。** 已有匹配检查点可直接用于迁移和推理。若从授权数据训练，以下命令按发布的词表在新的数据目录准备该子集，再用训练与验证划分训练。200词脚本的相对路径从 `nationalcsl200_stage_v1/` 解析。

```bash
python nationalcsl200_stage_v1/prepare_nationalcsl200.py --dataset-root ../local_data/NationalCSL-DP --class-list nationalcsl200_v2/class_order.csv --output ../local_data/prepared200
python nationalcsl200_stage_v1/train_nationalcsl200.py --data-root ../local_data/prepared200 --stage tune --device cpu
```

**200词离线推理。** 将自己的一段孤立词视频或一个原始 V2 特征文件放入仓库根目录下的 `local_data/`。视频输入需 Hand Landmarker；特征输入仅需匹配的 Transformer 检查点。

```bash
python nationalcsl200_stage_v1/infer_nationalcsl200.py --video ../local_data/clip.mp4 --device cpu --top-k 3
python nationalcsl200_stage_v1/infer_nationalcsl200.py --feature ../local_data/clip.npy --device cpu --top-k 3
```

**摄像头演示。** 第一条命令仅预览彩色摄像头。第二条命令默认加载200词模型，可以录制一段手语或打开已有视频。用户手动标记动作起止，完成后界面显示 Top-3、置信度和处理耗时。

```bash
python domain_mvp_v1/realtime_sign_demo.py --preview-only
python domain_mvp_v1/realtime_sign_demo.py
```

界面实际连接两个后端：默认的自定义200词模型，以及可切换的医疗11词模式。后者在旧领域20词模型的指定 11 个输出上计算概率，低于 0.40 时显示“未识别，请重试”。日常20词模型目前仅用于离线训练和验证。演示的识别单位是手动切分的单个词。[摄像头后端](domain_mvp_v1/realtime_sign_engine.py) · [医疗模式配置](domain_mvp_v1/medical_mode_config.json)

## 数据与署名

- **NationalCSL-DP** 由乐山师范学院特殊教育语言智能四川省哲学社会科学重点实验室等研究者建设。数据论文作者为 Peng Jin、Hongkai Li、Jun Yang、Yazhou Ren、Yuhao Li、Lilan Zhou、Jin Liu、Mei Zhang、Xiaorong Pu、Siyuan Jing。此处的词表保留原 gloss ID，并另行选词、排序和增加场景显示名。200词是本项目的自定义子集。数据来源、使用条款和原作者署名见 [Scientific Data 论文](https://doi.org/10.1038/s41597-025-04986-x)、[Figshare 数据页](https://figshare.com/articles/media/NationalCSL-DP/27261843) 和 [实验室发布说明](https://lise.lsnu.edu.cn/kxyj/dbxcg2.htm)。
- **MediaPipe Hand Landmarker** 由 Google 提供；本项目使用 Tasks API 提取关键点。模型下载和模型卡见 [Google 官方文档](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)。训练和推理使用 PyTorch，摄像头采集使用 OpenCV。
