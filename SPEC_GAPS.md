# 实现假设与稿件缺口

本文件是本仓库最重要的部分：它说明**哪些地方论文没写清、我按什么实现、你怎么改**。
任何"照论文重写"的人都必须面对这几处；审稿人（尤其 Reviewer 1 第 6 条）问的正是这些。

## 一、稿件本身缺定义的地方（4 处，需作者确认）

### G1. 式 (9) 三个尺度用了同一个权重符号

- **稿件原文**：`Pi = (G1⊗Gω) ⊕ (G2⊗Gω) ⊕ (G3⊗Gω)`
- **问题**：三个尺度乘的是**同一个** `ω`；而正文 3.4 节写 "The adaptive attention weight ▢ is obtained.
  Thirdly, the ▢ are multiplied with G1, G2 and G3" —— **权重符号是空的**。三种尺度的 α/β/γ 如何产生，论文没有定义。
- **本实现**（`model.py::AF`）：拼接 G1..G3 → 1×1 conv 输出 3 通道 → softmax → 三个权重逐尺度相乘后求和。
- **若要改**：把 `AF.mix` 换成你实际用的权重生成方式即可，其余不动。
- **影响**：表 2 Exp.4、表 7 Ours 行；若权重生成不同，MFIEM 的增益会变。

### G2. 式 (7) 的 `Spilt`

- **稿件原文**：`ωp, ωk = Spilt(C3×3BR(C3×3BR(fpk)))`
- **问题**：`Spilt` 是 `Split` 的拼写错误；且没有写清"输出 2 通道后如何拆"。
- **本实现**（`FSDAFM`）：两个 C3×3BR 后接 1 通道 1×1 conv 得 2 通道，`softmax(dim=1)` 后取通道 0/1 为 ωp、ωk
  （保证 ωp+ωk=1）。若原文用 sigmoid 独立加权，改成 `sigmoid` 即可。
- **影响**：FSD-AFM 的融合行为，进而影响表 2 Exp.3、表 6 Exp.2。

### G3. 式 (11) 的 AP 分母疑似写错

- **稿件原文**：`AP = (1/t)Σ(1/th)Σ TP_t/(TP_t + FN_t)`
- **问题**：`TP/(TP+FN)` 是 **recall**；AP 的逐点 precision 应为 `TP/(TP+FP)`。按原式，AP 与 AR 会算出同一个量。
- **本实现**（`metrics.py`）：严格按 COCO 定义（precision = TP/(TP+FP)，101 点插值）。
- **必须做**：核对式 (11) 是否漏印了 FP；若原稿确实如此，这是**指标定义错误**，审稿人一旦发现会直接质疑全部数字。
- **影响**：全篇六项指标的定义基础。

### G4. As2M 未定义

- **稿件原文**：图 8 标题为 "Overall structure of MSHM"，3.1 节说 MFIEM 由 "ECA module, AF_i module and MSHM module" 组成，
  3.4 节 Stage 3 却直接写 "In As2M_i(i∈{1,2,3})" —— **As2M 首次出现即使用，全篇无定义**，与 MSHM 的关系也未说明。
- **本实现**：把 As2M 视为 MSHM 的子模块（见 `As2M` 类），MSHM = {As2M_1..3 + 拼接 + ECA + 1×1}。
- **必须做**：正文补一句定义（如 "MSHM consists of three As2M blocks …"）。

## 二、本实现的 6 处工程假设（可配置）

| # | 位置 | 论文表述 | 本实现的默认 | 如何改 |
|---|---|---|---|---|
| **A1** | EWT_block | 2DWT 后各子带为原图 1/2 分辨率，但未说如何与空间分支对齐 | 1×1 conv 后用 `nearest` 上采样 ×2 | `EWTBlock.forward` 中的 `F.interpolate` |
| **A2** | 双分支骨干 | 3.1 节："每个分支每级各一个模块"，但未说两支如何合并 | 每级逐元素相加；C5 由 FSD-AFM 融合 | `TwoBranchBackbone.forward`（改为 concat 需同步 1×1 降维） |
| **A3** | MFIEM / AF_i | 只说 "upsample or downsample"，比例散落在文字里 | AF_1 (×2,×2,×1)、AF_2 (×0.5,×1,×2)、AF_3 (×0.25,×0.5,×1) | `MFIEM.__init__` 的 `factors` |
| **A4** | 式 (9) | 见 G1 | 3 通道 softmax | 见 G1 |
| **A5** | As2M | "X'Ci and X'Pi are matrix multiplied" | Hadamard 逐元素积 | `As2M(matmul=True)`（需自行补 matmul 分支） |
| **A6** | 预测头 | 未描述锚点设置 | 3 个长宽比 (1,1)/(1,2)/(2,1)，每级每位置 3 锚 | `PredictionHead(num_priors=...)` + `make_priors(aspect=...)` |

## 三、评测口径必须与你原脚本对齐

`metrics.py` 无 pycocotools 依赖，按 COCO 定义实现：AP 用 101 点插值、AR 取 maxDets=100 时的 recall、
`mAP` 为 IoU 0.50:0.95 十档平均、`AP` 为单档 0.75。

**但**：若你原来的脚本用的是 VOC 11 点插值、或 AR 的 maxDets 不同、或 mask 按 0.5 二值化的时机不同，
数字会系统性偏差 1–3 点。**必须先做第 8 节第 1 步的基线对齐**（`exp1_yolact` 一折 vs 论文 YOLACT 行），
差 >1.5 点就不要继续，先对齐评测。

## 四、尚未实现的部分（有意留空）

| 项目 | 原因 |
|---|---|
| 训练增强（mosaic / HSV / 随机缩放） | 论文未写增强细节；`data.py` 只有水平翻转。补增强会改变基线，必须**所有模型一致** |
| 学习率 warmup / 多步衰减 | 论文只给 lr=0.001、SGD、momentum 0.9、wd 5e-4，未给策略；本实现用 cosine |
| 多卡 / 混合精度 | 作者环境为单卡 TITAN V、FP32 |
| YOLO11-seg 等外部基线 | 属另一个仓库（ultralytics），见 `yolo11seg-integration.md` |
| 脑肿瘤（BraTS）数据管线 | 需要 NIfTI 读取与切片策略，且现稿对该数据集的描述与事实不符，先修文字再谈代码 |
