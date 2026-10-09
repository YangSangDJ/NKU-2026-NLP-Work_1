# 作业一：文本分类实现 —— 代码运行说明

NYT 新闻分类实验：比较词袋（Bag of Words）、词向量（GloVe / Word2Vec）与预训练语言模型（BERT）三类文本表示方法，统一在 NYT 测试集上报告 **Accuracy** 与 **Macro-F1**。

> **关于本仓库：** 因体积限制，仓库中**不含**课程数据集（`HW-1/*.csv`）、预训练 GloVe 文件
> （`HW-1/glove.6B.100d.txt`，331 MB）、BERT 权重与训练缓存（`code/artifacts/`）、以及数据划分缓存
> （`code/split/`）。这些文件均可按下方「三、数据准备」获取，或由代码自动生成
> （划分使用固定种子 42，重新生成的结果与缓存的划分完全一致）。详见 `.gitignore`。
>
> 实验报告以 PDF 形式提交，见 `NLP第一次作业-2313943-杨桑多杰.pdf`；报告的 LaTeX 源码
> （`Report/`）与配图为本地产物，未纳入仓库。

## 一、项目结构

```
Lab1/
├── NLP第一次作业-2313943-杨桑多杰.pdf   # 实验报告（PDF）
├── README.md                    # 本运行说明
├── HW-1/                        # 数据目录（作业提供）
│   ├── nyt.csv                  # NYT 新闻数据集（text, label），11,519 条
│   ├── ag.csv                   # AG News 文本（text），90,000 条，用于训练 Word2Vec
│   └── glove.6B.100d.txt        # 预训练 GloVe 100 维词向量（需自行下载，见下文）
├── code/                        # 完整实验代码
│   ├── common.py                # 数据加载、80/10/10 划分、分词、评价指标（所有实验共用）
│   ├── task1_bow.py             # Task 1：Binary BoW / Word Frequency
│   ├── task2_glove.py           # Task 2：预训练 GloVe 100d + LR
│   ├── task2_word2vec.py        # Task 2：AG News / NYT 上训练 Word2Vec + LR
│   ├── task3_bert.py            # Task 3：BERT 微调（max_length=64，3 epochs）
│   ├── run_all.py               # 一键运行全部实验
│   ├── make_figure.py           # 生成结果对比图（Report/images/results.png）
│   ├── analyze_truncation.py    # （分析用）BERT 输入截断统计
│   ├── download.py              # （工具）带重试/断点续传的下载脚本
│   ├── download_bert_files.py   # （工具）预下载 BERT 权重
│   ├── requirements.txt         # Python 依赖
│   ├── split/                   # 80/10/10 划分缓存（train/valid/test.csv）
│   ├── results/                 # 各模型测试集指标（JSON）
│   └── artifacts/               # 本地 BERT 权重缓存（可删除后重新下载）
└── Report/                      # 报告 LaTeX 源码与配图（本地产物，未纳入仓库）
```

## 二、环境要求

- Python **3.12**（注意：gensim 目前没有 Python 3.14 的预编译 wheel，建议使用 3.12）
- 依赖安装：

```bash
pip install -r code/requirements.txt
```

依赖清单：`pandas`、`numpy`、`scikit-learn`、`nltk`、`gensim`、`torch`、`transformers`
（另有 `matplotlib`，仅 `make_figure.py` 生成结果对比图时需要）。
BERT 微调建议在有 NVIDIA GPU + CUDA 版 PyTorch 的环境运行（CPU 也可运行但较慢）。

nltk 的 `punkt_tab` 数据（`nltk.download('punkt_tab')`）若下载失败，代码会自动回退到等价的正则分词，不影响运行。

## 三、数据准备

1. 将作业提供的 `HW-1/nyt.csv`、`HW-1/ag.csv` 放在 `HW-1/` 目录（这两个文件未纳入仓库，需从课程渠道获取）。
2. **GloVe 100 维词向量**：下载 `glove.6B.100d.txt` 放入 `HW-1/`。
   - 官方地址：<http://nlp.stanford.edu/data/glove.6B.zip>（解压后取 `glove.6B.100d.txt`）
   - 国内镜像示例：<https://hf-mirror.com/datasets/dodekVanBurak/glove6B100d/resolve/main/glove.6B.100d.txt>
3. **BERT 权重**：脚本默认从 `code/artifacts/bert-base-uncased/` 加载；
   若该目录不存在，则自动从 Hugging Face 下载 `google-bert/bert-base-uncased`。
   国内网络可先设置镜像后预下载：

```bash
# Windows PowerShell
$env:HF_ENDPOINT = 'https://hf-mirror.com'
python code/download_bert_files.py
```

## 四、运行实验

在 `code/` 目录下执行（推荐；脚本内部按自身位置解析数据路径，从其他目录调用同样可行）：

```bash
cd code

# 一键运行全部实验（Task 1 → 2 → 3）
python run_all.py

# 或逐步运行
python task1_bow.py          # Task 1：词袋
python task2_glove.py        # Task 2：GloVe
python task2_word2vec.py ag  # Task 2：AG News 训练的 Word2Vec
python task2_word2vec.py nyt # Task 2：NYT 训练的 Word2Vec
python task3_bert.py         # Task 3：BERT 微调
```

> 说明：`task2_word2vec.py` 会把训练好的 Word2Vec 模型缓存到 `code/artifacts/`，
> 再次运行直接复用；如需重新训练请删除对应 `.model` 文件。

## 五、实验设置（与作业要求一致）

| 项目 | 设置 |
|---|---|
| 数据划分 | NYT 整体随机打乱，80% 训练 / 10% 验证 / 10% 测试（种子 42，分层抽样），划分缓存于 `code/split/`，所有实验共用 |
| 分词 | 小写 + `nltk.word_tokenize`，仅保留长度≥2 的字母词元；词表只由训练集构建 |
| 评价指标 | Accuracy、Macro-F1（测试集上报告） |
| Task 1 | Binary BoW / Word Frequency，One-vs-Rest Logistic Regression（liblinear，`C` 由验证集 Macro-F1 选择） |
| Task 2 | 100 维词向量（GloVe / AG News Word2Vec / NYT Word2Vec），文档向量=词向量平均，StandardScaler + LR |
| Task 3 | `google-bert/bert-base-uncased`，`max_length=64`，3 epochs，AdamW lr=2e-5（warmup+线性衰减），batch size 32 |

## 六、结果

各模型的测试集指标自动写入 `code/results/*.json`，例如：

| 方法 | Accuracy | Macro-F1 |
|---|---|---|
| Binary Bag of Words + LR | 0.9887 | 0.9727 |
| Word Frequency + LR | 0.9896 | 0.9743 |
| GloVe 100d + LR | 0.9852 | 0.9646 |
| Word2Vec (AG News) + LR | 0.9852 | 0.9637 |
| Word2Vec (NYT) + LR | 0.9922 | 0.9792 |
| BERT-base-uncased（微调） | 0.9818 | 0.9612 |

> 注：由于随机种子固定（42）且数据划分一致，上述结果可复现；不同环境因 gensim 训练
> 并行等微小差异可能有合理波动，不影响结论。

生成报告中的结果对比图：

```bash
cd code && python make_figure.py   # 输出 Report/images/results.png
```

实验报告（PDF）见 `NLP第一次作业-2313943-杨桑多杰.pdf`；其 LaTeX 源码位于本地的 `Report/` 目录
（`Report/main.tex`），未纳入仓库。
