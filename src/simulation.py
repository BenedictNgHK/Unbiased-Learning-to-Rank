import numpy as np
from sentence_transformers import CrossEncoder
import src.config as config

class UserSimulator:
    """
    负责：
    - 计算 position bias（propensity）
    - 通过 Oracle Cross-Encoder 估计相关性概率（作为点击倾向）
    - 按 seed 生成可复现的点击日志
    """
    def __init__(self):
        self.oracle_name = getattr(config, "ORACLE_ENCODER_NAME", "cross-encoder/ms-marco-MiniLM-L-12-v2")
        self.max_len_ce = getattr(config, "MAX_LEN_CE", 256)
        # 作为“真值打分器”（仅用于离线评估/模拟）
        self.oracle = CrossEncoder(self.oracle_name, max_length=self.max_len_ce)

    def get_propensities(self, k: int):
        """
        位置曝光概率：P(examine@rank=r) = 1/sqrt(r)
        """
        ranks = np.arange(1, k + 1, dtype=float)
        return 1.0 / np.sqrt(ranks)

    def get_relevance_scores(self, query: str, documents):
        """
        用 Oracle CE 得到相关性得分 -> Sigmoid 概率（0~1）
        返回: {doc_text: prob}
        """
        if not documents:
            return {}
        pairs = [[query, d] for d in documents]
        scores = np.array(self.oracle.predict(pairs), dtype=float)
        probs = 1.0 / (1.0 + np.exp(-scores))  # Sigmoid 到 [0,1]
        return {doc: float(p) for doc, p in zip(documents, probs)}

    def simulate_clicks(self, query: str, documents, k: int = None, seed: int = None):
        """
        生成点击日志（可选固定 seed 以复现实验）：
        P(click) = P(examine@rank) * P(relevance)
        """
        if k is None:
            k = len(documents)
        documents = list(documents)[:k]

        rng = np.random.default_rng(seed) if seed is not None else None

        rel_map = self.get_relevance_scores(query, documents)
        relevance_probs = [rel_map[d] for d in documents]
        propensities = self.get_propensities(len(documents))

        logs = []
        for rank, (doc, rel, prop) in enumerate(zip(documents, relevance_probs, propensities), start=1):
            click_prob = float(rel * prop)
            if rng is not None:
                clicked = rng.random() < click_prob
            else:
                clicked = np.random.rand() < click_prob
            logs.append({
                "query": query,
                "doc_text": doc,
                "rank": rank,
                "relevance_prob": float(rel),
                "propensity": float(prop),
                "click": 1 if clicked else 0
            })
        return logs