# TWF²-YOLACT — reference code

按 Manuscript 第 3 节实现的 TWF²-YOLACT：**2DWT-DFEM**（EWT_block）、**FSD-AFM**、**MFIEM**，外加 YOLACT 式
FPN / Protonet / 预测头，以及 5 折训练、六项指标评测、效率测量与统计汇总脚本。

> ## ⚠️ 先说清楚这是什么、不是什么
>
> **是**：一份按论文描述重写的、可训练可评测的实现，用来补 5 折交叉验证、效率表、ECA 方向消融。
>
> **不是**：产出论文表 2/4/6/7/8 那份数字的原始代码。两者不是同一个二进制实现，**数字不会自动相同**。
> Scientific Reports 的 Zenodo 要求是"产生论文结果的代码"；若把本仓库当原代码提交，属于错误陈述。
> 本仓库的正确用途：(1) 补充实验；(2) 在原始代码遗失时的重建基线；(3) 公开实现细节以支持可复现性。

## 1. 安装

```bash
pip install -r requirements.txt      # torch>=1.12 / torchvision / numpy / pillow / pyyaml / thop(optional)
```

论文环境为 PyTorch 1.12.1 + CUDA 11.3 + TITAN V；更新的版本应当也能跑。

## 2. 数据准备

支持 **LabelMe 多边形 JSON**（即论文 4.1 节的标注格式），无需转换。

```bash
# 生成 list.txt：每行 "<image_path> <labelme_json>"
python build_list.py --images /path/to/png --jsons /path/to/json --out data/list.txt

# 生成 5 折划分（分层：按每图实例数 0-1 / 2-3 / >=4）
python make_folds.py --list-file data/list.txt --k 5 --seed 42 --out data/folds.json
```

`data/folds.json` 含每折 train/val 列表与 `fingerprint`；**所有模型必须共用同一份**（审稿意见 R1.6 的核心）。
该文件也是应当随代码一起发布的划分文件。

## 3. 快速自检（不需要 GPU）

```bash
python smoke_test.py
```

检查：Haar DWT/IWT 往返误差、四种变体前向、锚点与 mask 系数对齐、postprocess 可运行。
**先让这一步通过，再开 40 次训练**——否则 180 GPU·小时会在第 3 次运行时全部作废。

## 4. 训练 / 评测

```bash
# 单折
python train.py --config configs/exp4_full.yaml --fold 0 --out runs/full_f0

# 全部（配置 × 折），已完成自动跳过，可断点续跑
python run_all.py --configs configs/exp*.yaml --folds 5 --out runs

# 单独评测某权重
python evaluate.py --config configs/exp4_full.yaml --weights runs/full_f0/best.pth \
    --list-file data/val_f0.txt
```

每次运行写出 `runs/<name>/metrics.json`（六项指标 + params + wall time）与 `last.pth`。

## 5. 配置与消融的对应关系

| 配置 | 对应论文 | 说明 |
|---|---|---|
| `exp1_yolact.yaml` | 表 2 Exp.1 / 表 6 Exp.1 / 表 7 YOLACT 行 | 纯 YOLACT（ResNet50+FPN+头） |
| `exp2_dfem.yaml` | 表 2 Exp.2 / 表 4 Exp.7 | 加 2DWT-DFEM，无 FSD-AFM |
| `exp3_dfem_fsd.yaml` | 表 2 Exp.3 / **表 6 Exp.2** | 再加 FSD-AFM（最终模型的骨干阶段） |
| `exp4_full.yaml` | 表 2 Exp.4 / 表 7 Ours | 再加 MFIEM = 完整模型 |
| `eca_dh_only.yaml` | 表 4 Exp.2 | ECA 只作用于 Dh |
| `eca_dh_dw_dd.yaml` | 表 4 Exp.8 | 三方向全开 |
| `ewt_iwt.yaml` | 表 6 Exp.3 | 通道拼接换成逆小波重建 |

> 复用关系（省算力的关键）：`exp1` 一次训练同时填三张表的基线行；`exp3` 同时填表 2 Exp.3 与表 6 Exp.2。

## 6. 效率表（R1.3 / R2.3）

```bash
python profile.py --config configs/exp4_full.yaml --out eff_full.json
python profile.py --config configs/exp1_yolact.yaml --out eff_yolact.json
```

输出六字段：硬件、batch、精度、是否含前后处理与 NMS、计时方式、输入分辨率；另有 params / FLOPs /
latency / FPS。`FLOPs` 需要 `thop`，未安装时为 `null`。

## 7. 统计（R2.4）

```bash
python stats.py --runs runs --group-by variant --compare full dfem_fsd
```

打印 `mean ± std`（ddof=1）与配对 Wilcoxon 精确 p。

> **n=5 的硬约束**：5 对样本的精确双侧 Wilcoxon 最小 p 就是 **0.0625**，永远到不了 0.05。
> 因此正文只能写 `p = 0.0625 (exact, n = 5)`，不能写 `p < 0.05`。要 p<0.05 必须 n≥6。

## 8. 建议的复现顺序

1. `exp1_yolact` 跑 1 折 → 与论文表 7 的 YOLACT 行（86.0/89.6/73.9/75.0/68.9/70.8）对比。
   **差 >1.5 点说明评测脚本或数据管线与原始实现不一致，先别继续。**
2. 对齐后再跑 `exp2 → exp3 → exp4`，每步核对增量方向是否正确（应为正）。
3. 最后才铺 5 折全量。

## 9. 论文未定义处的处理

见 [SPEC_GAPS.md](SPEC_GAPS.md)。共 6 处实现假设（A1–A6）+ 4 处稿件本身的缺陷
（式 9 重复权重、式 7 `Spilt`、式 11 AP 分母、As2M 未定义）。每一处都写明了"论文原文 / 本实现 /
如何改"。若审稿人追问架构细节，这份文件就是答案来源。

## 10. 目录

```
twf2yolact/model.py     所有模块：ECA / Haar / EWTBlock / FSDAFM / AF / As2M / MFIEM / 骨干 / FPN / 头 / 后处理
twf2yolact/data.py      LabelMe 解析、数据集、collate
twf2yolact/losses.py    box / cls / mask 三项损失
twf2yolact/metrics.py   COCO 风格六指标（无 pycocotools 依赖）
twf2yolact/engine.py    模型构建、单轮训练、评测循环
train.py evaluate.py profile.py stats.py make_folds.py build_list.py run_all.py smoke_test.py
configs/                消融配置
```
