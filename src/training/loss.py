import torch
import torch.nn as nn

class DPRLoss(nn.Module):
    def forward(self, queries, documents, pos_mask):
        """
        @param queries:   (B_q, D) query embeddings
        @param documents: (B_d, D) document embeddings
        @param pos_mask:  (B_q, B_d) bool, True where document j is relevant to query i

        Each positive is scored against the query's negatives only, so several
        positives for one query are all treated as correct at the same time.
        Returns: scalar loss (mean over all positive pairs)
        """
        scores = queries @ documents.T                                       # (B_q, B_d)
        pos_mask = pos_mask.to(scores.device)

        neg_scores = scores.masked_fill(pos_mask, float('-inf'))             # hide positives
        log_neg = torch.logsumexp(neg_scores, dim=1, keepdim=True)           # log sum_neg exp(s)

        # -log( exp(s_ij) / (exp(s_ij) + sum_neg exp(s_ik)) )
        per_pair = torch.logaddexp(scores, log_neg) - scores                 # (B_q, B_d)
        return per_pair[pos_mask].mean()

# import torch.nn.functional as F

# class DPRLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, queries: torch.tensor, documents: torch.tensor):
#         """
#         @param queries: batch_size x embedding_dim
#         @param documents: batch_size x embedding_dim

#         Computes cross entropy between the query 

#         Returns: A 1d torch tensor 
#         """


#         # Entry [i, j] is the dot product between query i and document j: 
#         # how relevant the model thinks document j is to query i
#         score_matrix = queries @ documents.T                             # (B_q, B_d)
#         # the correct document for query i is document i. (IS THIS WRONG? YES)
#         targets = torch.arange(queries.size(0), device=queries.device)   # positives on the diagonal
        
#         # softmax turns the scores into probabilities over the 50 documents
#         loss =  F.cross_entropy(score_matrix, targets)

#         return loss

#         # raise NotImplementedError()
