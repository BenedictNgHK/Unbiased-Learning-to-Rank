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
        
        target_ranking: List of documents (strings) in the new order.
        click_logs: List of dicts {doc_text, click, propensity, ...} from the logging policy.
        
        SNIPS = (Sum_i (Y_i * w_i)) / (Sum_i w_i)
        where Y_i is the reward (click) and w_i is the importance weight.
        w_i = pi_target(d|q) / pi_logging(d|q)
        
        In a deterministic ranking setting:
        pi_target(d|q) = 1 if d is at rank k (or just present?), 0 otherwise.
        However, standard IPS for ranking usually treats the 'action' as the list or item at k.
        
        Simplified Counterfactual Evaluation for Ranking (Item-level):
        We want to estimate the number of relevant items retrieved or DCG.
        
        Let's assume we are estimating the "Total Clicks" we would get.
        For each position k in Target Ranking:
            Find the document d at target_k.
            Look for d in the Click Logs.
            If d appeared in logs at rank j:
                Y = click_in_log
                Propensity = p_j (propensity at rank j in logs)
                Weight = 1 / p_j (since target places it at k, we assume target examines it with prob 1? 
                          Or better, we adjust for the target position bias?
                          
        Standard approach for Position Bias correction (Propensity Scoring):
        R_hat = Sum_{d in logs} (Click_d / Propensity_d) * Indicator(d in Target)
        
        If we want to estimate the relevance of the Target List:
        We only sum up the inverse-propensity-weighted clicks for documents that appear in our Target List.
        
        SNIPS normalization:
        Denominator = Sum_{d in logs} (1 / Propensity_d) * Indicator(d in Target)
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
    def calculate_dcg(target_ranking, click_logs):
        """
        Calculates DCG using the simulated 'true' relevance probabilities from logs if available,
        or just using the clicks as binary relevance.
        """
        # Build a lookup for relevance
        # In simulation, we have 'relevance_prob' which is the ground truth.
        # We can use that to compute the 'Ideal' metric for reference.
        doc_rel_map = {log['doc_text']: log.get('relevance_prob', 0) for log in click_logs}
        
        dcg = 0.0
        for i, doc in enumerate(target_ranking):
            rel = doc_rel_map.get(doc, 0)
            rank = i + 1
            dcg += rel / np.log2(rank + 1)
            
        return dcg
