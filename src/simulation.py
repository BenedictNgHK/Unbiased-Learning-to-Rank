import numpy as np
from sentence_transformers import CrossEncoder
import src.config as config

class UserSimulator:
    """
    Simulates user clicks based on a Position-Based Model (PBM).
    P(Click = 1 | u, d, r) = P(Examine = 1 | r) * P(Relevant = 1 | u, d)
    """
    def __init__(self, ground_truth_model_name=config.CROSS_ENCODER_NAME):
        # We use a Cross-Encoder as the "Oracle" / "Ground Truth" for relevance
        # In a real scenario, this would be unknown or latent.
        print(f"Loading Simulator Ground Truth Model: {ground_truth_model_name}...")
        self.ground_truth_model = CrossEncoder(ground_truth_model_name)
        
    def get_relevance_scores(self, query, documents):
        """
        Returns the true relevance probabilities P(Relevant | q, d) for a list of documents.
        """
        if not documents:
            return {}
            
        pairs = [[query, doc] for doc in documents]
        scores = self.ground_truth_model.predict(pairs)
        probs = 1 / (1 + np.exp(-scores))
        
        return {doc: float(prob) for doc, prob in zip(documents, probs)}

    def get_propensities(self, k, power=config.DEFAULT_POSITION_BIAS_POWER):
        """
        Returns position bias probabilities P(E=1 | r) for ranks 1..k
        Simple decay model: 1 / (rank + 1)^power
        """
        ranks = np.arange(1, k + 1)
        # Use a slightly milder decay than 1/r to make it interesting
        propensities = 1.0 / np.power(ranks, power)
        return propensities

    def simulate_clicks(self, query, documents, k=None):
        """
        Simulates clicks for a given query and a list of documents.
        Returns a list of dictionaries containing click information.
        """
        if k is None:
            k = len(documents)
        
        documents = documents[:k]
        
        # 1. Calculate True Relevance Probability (Attractiveness)
        rel_map = self.get_relevance_scores(query, documents)
        relevance_probs = [rel_map[doc] for doc in documents]
        
        # 2. Calculate Examination Probability (Position Bias)
        propensities = self.get_propensities(len(documents))
        
        # 3. Simulate Clicks
        click_logs = []
        
        for rank, (doc, rel_prob, prop) in enumerate(zip(documents, relevance_probs, propensities)):
            click_prob = rel_prob * prop
            is_clicked = np.random.rand() < click_prob
            
            click_logs.append({
                "query": query,
                "doc_text": doc,
                "rank": rank + 1,
                "relevance_prob": float(rel_prob),
                "propensity": float(prop),
                "click": 1 if is_clicked else 0
            })
            
        return click_logs
