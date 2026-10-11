# 基于多维度特征融合相似度测度的中文新闻文本去重与聚类方法研究

> 毕业论文（模拟稿）配套开源仓库 · Graduation Thesis (Simulation) Repository

本仓库收录一篇 6000 字以上的毕业论文模拟稿，以及其全部可复现材料。论文**优先复用已学的「多维度特征融合文本相似度测度方法」**，并将其迁移到「中文新闻文本去重与聚类」这一感兴趣的研究方向上，研究步骤完整、清晰、合理。

## 研究简介

面对新闻"同题异构"带来的海量近似重复文本，本文提出一种**统计—句法—语义三维互补**的相似度测度与融合框架：

- **统计视图**：TF-IDF 向量（标题词 ×3 位置权重）
- **句法视图**：词序 n-gram 多重集相似度
- **语义视图**：字符级嵌入余弦相似度
- **融合**：方差自适应加权（无监督）/ 网格搜索加权（有监督）

下游完成两个任务：**新闻去重**（成对 P/R/F1）与**事件聚类**（单连接聚类 + ARI/NMI/BCubed-F1），并通过消融实验验证各维度贡献。

## 实验结果（摘要）

在自建 33 篇、覆盖 23 个真实新闻事件的数据集上：

| 方法 | 去重 F1 | 聚类 ARI | 聚类 NMI | 聚类 BCubed-F1 |
|---|---|---|---|---|
| A. 统计基线 (TF-IDF) | 71.0% | 0.676 | 0.948 | 0.838 |
| B. 语义基线 | 82.8% | 0.796 | 0.977 | 0.925 |
| C. 自适应融合 | 71.0% | 0.676 | 0.948 | 0.838 |
| **D. 网格优化融合 (本文)** | **85.7%** | **0.854** | **0.984** | **0.944** |

消融实验表明语义维度起主导作用（去除后 F1 骤降至 4.4%），但统计与句法维度在"字面相似、事件不同"的陷阱样本上提供关键判别，使完整融合在聚类综合指标上取得最高值。

> 完整表格、阈值扫描与参数敏感性分析见 `thesis.md` 第四章，原始指标见 `results/experiment_results.json`。

## 目录结构

```
news-dedup-thesis/
├── thesis.md                      # 毕业论文模拟稿（6000+ 字，含中英文摘要、六章正文、参考文献、致谢）
├── README.md                      # 本文件
├── requirements.txt              # Python 依赖（仅 numpy）
├── LICENSE                       # MIT License
├── .gitignore
├── src/
│   └── news_similarity.py        # 多维度融合相似度引擎（统计/句法/语义 + 自适应/网格加权）
├── data/
│   └── news_sample.json          # 自建中文新闻数据集（33 篇 / 23 事件，含陷阱样本）
├── experiments/
│   └── run_news_experiment.py    # 对比实验 + 网格搜索 + 消融，输出指标
└── results/
    ├── run_out.txt               # 实验运行输出（可逐行核对）
    └── experiment_results.json   # 结构化实验结果
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行实验（生成 results/）
python experiments/run_news_experiment.py

# 3. 在代码中调用融合引擎
python -c "
from src.news_similarity import NewsSimilarityEngine
import json
docs = json.load(open('data/news_sample.json', encoding='utf-8'))
eng = NewsSimilarityEngine(docs)
M = eng.pairwise_matrix('fusion')
labels = eng.cluster(M, threshold=0.60)
print('簇数量:', len(set(labels)))
"
```

## 方法迁移说明

本文方法继承自 `thesis_repo`（电商商品文本相似度引擎）中已学的多维度特征融合测度，并针对中文新闻做了三点适配：

1. 引入**标题位置权重**（标题词 ×3），强化新闻核心语义载体；
2. 增加**句法视图**（词序 n-gram），专门应对"词表相近、结构不同"的样本；
3. 同时提供**无监督自适应加权**与**有监督网格搜索**两种权重方案，兼顾部署便利与性能上限。

## 许可证

[MIT License](./LICENSE) — 欢迎在注明出处的前提下自由复用、修改与二次分发。
