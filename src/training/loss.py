import torch
import torch.nn as nn

class DPRLoss(nn.Module):

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, queries, documents, pos_mask):
        """
        @param queries:   (B_q, D) query embeddings
        @param documents: (B_d, D) document embeddings
        @param pos_mask:  (B_q, B_d) bool, True where document j is relevant to query i

        Each positive is scored against the query's negatives only, so several
        positives for one query are all treated as correct at the same time.
        Returns: scalar loss (mean over all positive pairs)
        """
        # scores = queries @ documents.T                                       # (B_q, B_d)
        scores = queries @ documents.T / self.temperature
        pos_mask = pos_mask.to(scores.device)

        neg_scores = scores.masked_fill(pos_mask, float('-inf'))             # hide positives
        log_neg = torch.logsumexp(neg_scores, dim=1, keepdim=True)           # log sum_neg exp(s)

        # -log( exp(s_ij) / (exp(s_ij) + sum_neg exp(s_ik)) )
        per_pair = torch.logaddexp(scores, log_neg) - scores                 # (B_q, B_d)
        return per_pair[pos_mask].mean()
