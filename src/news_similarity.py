"""
news_similarity.py
==================
面向中文新闻文本的多维度特征融合相似度测度引擎（毕业论文配套实现）

本文在已学的「多维度特征融合文本相似度测度」方法基础上，将其迁移到
「中文新闻文本去重与聚类」任务。引擎沿用同一套核心方法论：

  1. 浅层统计特征 : 位置加权 TF-IDF + 类目/来源 One-Hot + 长度比辅助
  2. 中层句法特征 : 以词序熵、句数、平均句长等结构信号近似刻画句法信息
                    （依存句法解析器可插件式替换，见 _syntax_feature）
  3. 深层语义特征 : 字符/词 n-gram 稠密向量（若安装 sentence-transformers
                    则自动升级为 BERT 句向量）
  4. 融合策略     : 各子特征 min-max 归一化后，按「两两相似度分布离散度」
                    自适应赋权，再做加权融合（与论文方法完全一致）

仅依赖 numpy；jieba 为可选（自动回退到字符级切分）。
运行：python experiments/run_news_experiment.py
"""

import json
import math
import re
import numpy as np

try:
    import jieba
    jieba.setLogLevel(20)
    _HAS_JIEBA = True
except Exception:
    _HAS_JIEBA = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False


STOPWORDS = set(
    "的 了 和 与 及 是 在 我 你 他 她 它 这 那 有 对 为 也 都 把 被 让 给 着 到 从 上 下 中 内 里 "
    "很 更 最 吗 呢 啊 吧 哦 地 得 之 其 以 并 或 但 而 等 个 据 称 显示 表示 数据 介绍 引发 近日 据悉".split()
)


def tokenize(text: str):
    """分词：优先 jieba，回退到字符级 bigram。"""
    text = text or ""
    cn = "".join(re.findall(r"[\u4e00-\u9fa5]", text))
    en = re.findall(r"[a-z0-9]+", text.lower())
    tokens = []
    for i, ch in enumerate(cn):
        if ch not in STOPWORDS:
            tokens.append(ch)
        if i < len(cn) - 1:
            bg = cn[i] + cn[i + 1]
            if not any(s in bg for s in STOPWORDS if len(s) == 2):
                tokens.append(bg)
    if _HAS_JIEBA:
        for w in jieba.cut(text):
            w = w.strip().lower()
            if w and w not in STOPWORDS and re.search(r"[\u4e00-\u9fa5a-z0-9]", w):
                tokens.append(w)
    return tokens + en


class NewsSimilarityEngine:
    def __init__(self, news, bert_model=None):
        self.news = news
        self.docs = [self._text(n) for n in news]
        self.corpus = [tokenize(d) for d in self.docs]
        self.doc_count = len(self.corpus)

        # 全局词表与文档频率（固定维度 TF-IDF，保证任意两向量可计算余弦）
        self.vocab = sorted({t for d in self.corpus for t in d})
        self._df = {}
        for t in self.vocab:
            self._df[t] = sum(1 for d in self.corpus if t in d)

        # 类目 / 来源集合（用于浅层辅助 one-hot 特征）
        self.categories = sorted({n.get("category", "") for n in news})
        self.sources = sorted({n.get("source", "") for n in news})

        # BERT（可选）
        self.bert = None
        if bert_model and _HAS_ST:
            self.bert = SentenceTransformer(bert_model)

        # 预计算三类特征向量
        self.stat_v = [self._stat_feature(n, d) for n, d in zip(news, self.docs)]
        self.syn_v = [self._syntax_feature(n, d) for n, d in zip(news, self.docs)]
        self.sem_v = [self._semantic_feature(d) for d in self.docs]

        # 各维度 minmax 矩阵，便于复用
        self._mm = [
            [self._minmax(v) for v in self.stat_v],
            [self._minmax(v) for v in self.syn_v],
            [self._minmax(v) for v in self.sem_v],
        ]

        # 自适应融合权重：以各维度「两两相似度分布的离散度」衡量判别力
        # （与特征维数无关，避免高维语义向量被方差公式高估）
        N = len(news)
        self._sub_var = []
        for k in range(3):
            sims = [self._cosine(self._mm[k][i], self._mm[k][j])
                    for i in range(N) for j in range(i + 1, N)]
            self._sub_var.append(float(np.var(sims)) + 1e-9)

    # ---------- 文本构造 ----------
    @staticmethod
    def _text(n):
        return " ".join([
            n.get("title", ""), n.get("category", ""),
            n.get("content", ""), n.get("source", ""),
        ])

    # ---------- 浅层统计特征 ----------
    def _tfidf_vec(self, tokens, title_set=None):
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        L = max(len(tokens), 1)
        vec = np.zeros(len(self.vocab))
        for i, t in enumerate(self.vocab):
            if t in tf:
                idf = math.log((self.doc_count + 1) / (self._df[t] + 1)) + 1
                w = (tf[t] / L) * idf
                if title_set is not None and t in title_set:
                    w *= 3.0
                vec[i] = w
        return vec

    def _stat_feature(self, n, doc):
        # 标题词赋予更高位置权重（标题是新闻核心语义载体）
        toks = tokenize(doc)
        title_set = set(tokenize(n.get("title", "")))
        vec = self._tfidf_vec(toks, title_set=title_set)
        # 类目 / 来源 one-hot 浅层辅助信号
        aux = [0.0] * (len(self.categories) + len(self.sources))
        ci = self.categories.index(n.get("category", "")) if n.get("category", "") in self.categories else -1
        if ci >= 0:
            aux[ci] = 1.0
        si = self.sources.index(n.get("source", "")) if n.get("source", "") in self.sources else -1
        if si >= 0:
            aux[len(self.categories) + si] = 1.0
        # 标题/正文长度比
        tl = len(tokenize(n.get("title", "")))
        cl = max(len(toks), 1)
        aux.append(min(tl / cl, 1.0))
        return np.concatenate([vec, np.array(aux, dtype=float)])

    # ---------- 中层句法特征（简化结构特征，可插件式替换为 LTP 依存句法） ----------
    def _syntax_feature(self, n, doc):
        toks = tokenize(doc)
        L = max(len(toks), 1)
        from collections import Counter
        cnt = Counter(toks)
        ent = 0.0
        for c in cnt.values():
            pr = c / L
            ent -= pr * math.log(pr + 1e-9)
        # 句子数（以中文句末标点切分）
        sents = re.split(r"[。！？!?；;]", n.get("content", ""))
        sents = [s for s in sents if s.strip()]
        n_sent = max(len(sents), 1)
        avg_len = L / n_sent
        title_len = len(tokenize(n.get("title", "")))
        struct = [
            ent / math.log(L + 1),
            min(n_sent / 5.0, 1.0),
            min(avg_len / 20.0, 1.0),
            min(title_len / L, 1.0),
        ]
        return np.array(struct, dtype=float)

    # ---------- 深层语义特征 ----------
    def _semantic_feature(self, doc):
        if self.bert is not None:
            return self.bert.encode([doc])[0]
        # 回退：字符/词 n-gram 稠密向量（充当语义近似）
        toks = tokenize(doc)
        vec = self._tfidf_vec(toks)
        dim = 256
        emb = np.zeros(dim)
        for i, w in enumerate(vec):
            if w > 0:
                h = abs(hash(self.vocab[i])) % dim
                emb[h] += w
        nrm = np.linalg.norm(emb)
        if nrm > 0:
            emb = emb / nrm
        return emb

    # ---------- 工具 ----------
    @staticmethod
    def _minmax(v):
        v = np.array(v, dtype=float)
        lo, hi = v.min(), v.max()
        if hi - lo < 1e-9:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo)

    @staticmethod
    def _cosine(a, b):
        a, b = np.array(a, float), np.array(b, float)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # ---------- 相似度计算 ----------
    def similarity(self, i, j, weights=None):
        sv = [self._minmax(self.stat_v[i]), self._minmax(self.syn_v[i]), self._minmax(self.sem_v[i])]
        tv = [self._minmax(self.stat_v[j]), self._minmax(self.syn_v[j]), self._minmax(self.sem_v[j])]
        scores = [self._cosine(sv[k], tv[k]) for k in range(3)]
        if weights is not None:
            w = [x / (sum(weights) + 1e-9) for x in weights]
        else:
            tot = sum(self._sub_var) + 1e-9
            w = [v / tot for v in self._sub_var]
        return float(sum(w[k] * scores[k] for k in range(3)))

    # ---------- 单特征基线（用于对比实验） ----------
    def stat_only(self, i, j):
        return self._cosine(self._minmax(self.stat_v[i]), self._minmax(self.stat_v[j]))

    def sem_only(self, i, j):
        return self._cosine(self._minmax(self.sem_v[i]), self._minmax(self.sem_v[j]))

    def syn_only(self, i, j):
        return self._cosine(self._minmax(self.syn_v[i]), self._minmax(self.syn_v[j]))

    # ---------- 全量相似度矩阵（去重 / 聚类实验用） ----------
    def pairwise_matrix(self, mode="fusion"):
        N = len(self.news)
        M = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                if mode == "stat":
                    s = self.stat_only(i, j)
                elif mode == "sem":
                    s = self.sem_only(i, j)
                elif mode == "syn":
                    s = self.syn_only(i, j)
                else:
                    s = self.similarity(i, j)
                M[i, j] = M[j, i] = s
        return M


if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base, "data", "news_sample.json"), encoding="utf-8") as f:
        items = json.load(f)
    eng = NewsSimilarityEngine(items)
    print("引擎初始化完成，新闻数:", len(items))
    print("自适应融合权重 [统计, 句法, 语义] =", [round(x, 3) for x in eng._sub_var])
    M = eng.pairwise_matrix("fusion")
    print("相似度矩阵对角示例（最大非对角相似度）:")
    N = len(items)
    best = (0, 0, 0)
    for i in range(N):
        for j in range(i + 1, N):
            if M[i, j] > best[0]:
                best = (M[i, j], i, j)
    print("  最相似一对: %.3f  %s <-> %s" % (best[0], items[best[1]]["title"], items[best[2]]["title"]))
