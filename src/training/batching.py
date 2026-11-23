
import random
import torch

def build_batch(rows, max_pos=None):
    """
    @param rows: a slice of corpus.train (dict of lists)
    @param max_pos: if set, use at most this many randomly chosen positives per query (caps memory)
    Returns: query texts, unique document texts, (B_q, B_d) bool positive mask
    """
    queries_text = rows['question']
    unique_documents, unique_documents_text = [], []

    for i, doc_list in enumerate(rows['document_id']):
        idx = list(range(len(doc_list)))
        if max_pos is not None and len(idx) > max_pos:
            idx = random.sample(idx, max_pos)
        for j in idx:
            doc_id = doc_list[j]
            if doc_id not in unique_documents:
                unique_documents.append(doc_id)
                unique_documents_text.append(rows['text'][i][j])

    document_to_idx = {doc_id: k for k, doc_id in enumerate(unique_documents)}
    mask = torch.zeros((len(queries_text), len(unique_documents)), dtype=torch.bool)

    # mark every positive of query i that is present in the batch (sampled or not)
    for i, doc_list in enumerate(rows['document_id']):
        for doc_id in doc_list:
            if doc_id in document_to_idx:
                mask[i, document_to_idx[doc_id]] = True

    return queries_text, unique_documents_text, mask