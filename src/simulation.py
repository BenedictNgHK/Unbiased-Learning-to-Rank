# src/simulation.py  —— seed 可控版（整文件替换）
import os
import math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import src.config as config

def _to_device(m):
    if torch.cuda.is_available():
        return m.to("cuda")
    return m

class UserSimulator:
    """
    点击生成器（可控点击率 + 可选保证每个 query 至少有正/负样本）
    产出字段：rank, click, propensity, relevance_prob, doc_text
    """
    def __init__(
        self,
        click_temp: float = 0.75,         # 相似度→点击概率的温度（越小越“硬”）
        rel_floor: float = 0.05,          # 相关性下界，避免全 0
        rel_ceiling: float = 0.95,        # 相关性上界，避免全 1
        ensure_min_pos: int = 1,          # 每个 query 至少强制 1 个正样本（必要时）
        ensure_min_neg: int = 1,          # 每个 query 至少强制 1 个负样本（必要时）
        rng_seed: int | None = None,
        max_len: int | None = None
    ):
        self.click_temp = float(click_temp)
        self.rel_floor = float(rel_floor)
        self.rel_ceiling = float(rel_ceiling)
        self.ensure_min_pos = int(ensure_min_pos)
        self.ensure_min_neg = int(ensure_min_neg)
        # 默认内部 RNG（用于不传 seed 的情况）
        self.rng = np.random.RandomState(rng_seed if rng_seed is not None else 2025)
        self.max_len = max_len if max_len is not None else getattr(config, "MAX_LEN_CE", 256)

        oracle_name = getattr(config, "ORACLE_ENCODER_NAME", "cross-encoder/ms-marco-MiniLM-L-12-v2")
        self.tokenizer = AutoTokenizer.from_pretrained(oracle_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(oracle_name)
        self.model.eval()
        _to_device(self.model)

    # rank→检视概率：常见 position bias 曲线（1/√rank）
    @staticmethod
    def _propensity(rank: int) -> float:
        return 1.0 / math.sqrt(rank)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _map_scores_to_relprob(self, scores: np.ndarray) -> np.ndarray:
        """
        将原始 oracle 分数（logit）→ [0,1] 的相关性概率，带温度缩放并裁剪上下界。
        """
        if scores.size == 0:
            return scores
        # 归一：z-score + 温度
        mu = float(scores.mean())
        sd = float(scores.std()) if scores.std() > 1e-6 else 1.0
        z = (scores - mu) / sd
        rel = self._sigmoid(z / max(self.click_temp, 1e-6))
        rel = np.clip(rel, self.rel_floor, self.rel_ceiling)
        return rel

    @torch.inference_mode()
    def get_relevance_scores(self, query: str, docs: list[str]) -> dict[str, float]:
        if not docs:
            return {}
        pairs = [(query, d) for d in docs]
        enc = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        if torch.cuda.is_available():
            enc = {k: v.to("cuda") for k, v in enc.items()}
        logits = self.model(**enc).logits.squeeze(-1).detach().float().cpu().numpy()
        rel = self._map_scores_to_relprob(logits)
        return {d: float(r) for d, r in zip(docs, rel)}

    @torch.inference_mode()
    def simulate_clicks(self, query: str, docs: list[str], seed: int | None = None) -> list[dict]:
        """
        基于 position bias（propensity）× relevance_prob 生成点击。
        如全 0 或全 1，可按 ensure_min_pos/neg 最小化自救，避免训练 pair=0。
        seed: 若提供，则本次调用使用该 seed 的独立 RNG；不提供则使用实例的 self.rng。
        """
        if not docs:
            return []

        # 关键：每次请求可用独立 seed 控制随机性
        rng = np.random.RandomState(seed) if seed is not None else self.rng

        rel_map = self.get_relevance_scores(query, docs)
        rel_list = [rel_map.get(d, 0.5) for d in docs]

        logs = []
        clicks = []
        for i, d in enumerate(docs, start=1):
            p_exam = self._propensity(i)
            p_rel = rel_list[i-1]
            p_click = p_exam * p_rel
            c = int(rng.rand() < p_click)
            clicks.append(c)
            logs.append({
                "rank": i,
                "click": c,
                "propensity": float(p_exam),
                "relevance_prob": float(p_rel),
                "doc_text": d
            })

        # 最小平衡：如需要则保证至少 1 正 / 1 负（只在极端情况下触发）
        if self.ensure_min_pos > 0 and sum(clicks) == 0:
            idx = int(np.argmax(rel_list))
            logs[idx]["click"] = 1
        if self.ensure_min_neg > 0 and sum(1 - np.array(clicks)) == 0:
            idx = int(np.argmin(rel_list))
            logs[idx]["click"] = 0

        return logs