import numpy as np

class Evaluator:
    """
    Implements Offline Policy Evaluation (OPE) metrics.
    Focus on SNIPS (Self-Normalized Inverse Propensity Scoring).
    """
    
    @staticmethod
    def calculate_snips(target_ranking, click_logs):
        """
        Calculates the SNIPS estimate of the reward (relevance) for the target ranking,
        using historical click logs from a logging policy.
        """
        numerator = 0.0
        denominator = 0.0
        
        # We assume target_ranking is the list of text documents
        target_set = set(target_ranking)
        
        relevant_found_in_target = 0
        
        for log in click_logs:
            doc_text = log['doc_text']
            
            # If this document from the logs is also in our new Target Ranking
            if doc_text in target_set:
                click = log['click']
                propensity = log['propensity']
                
                # Avoid division by zero
                if propensity <= 0:
                    continue
                    
                weight = 1.0 / propensity
                
                numerator += (click * weight)
                denominator += weight
                
                if click:
                    relevant_found_in_target += 1
        
        if denominator == 0:
            return 0.0
            
        snips_score = numerator / denominator
        return snips_score

    @staticmethod
    def calculate_dcg(target_ranking, rel_map, k=None):
        """
        Calculates DCG@k given a ranking and a map of relevance scores.
        """
        if k is None:
            k = len(target_ranking)
            
        dcg = 0.0
        for i, doc in enumerate(target_ranking[:k]):
            rel = rel_map.get(doc, 0.0)
            rank = i + 1
            dcg += rel / np.log2(rank + 1)
            
        return dcg

    @staticmethod
    def calculate_ndcg(target_ranking, rel_map, k=None, idcg=None):
        """
        Calculates nDCG@k.
        If idcg is provided, uses it. 
        Otherwise, calculates IDCG based on sorting the relevance scores of the target_ranking (local optimality).
        """
        if k is None:
            k = len(target_ranking)
            
        dcg = Evaluator.calculate_dcg(target_ranking, rel_map, k)
        
        if idcg is None:
            # Calculate Ideal DCG based on available documents in this ranking
            # This measures "how well sorted is this specific list"
            # Note: For comparing two different lists (Baseline vs Target), 
            # you should provide a common IDCG (e.g. from the union of candidates).
            all_rels = sorted([rel_map.get(doc, 0.0) for doc in target_ranking], reverse=True)
            idcg = 0.0
            for i, rel in enumerate(all_rels[:k]):
                idcg += rel / np.log2((i + 1) + 1)
        
        if idcg == 0:
            return 0.0
            
        return dcg / idcg
