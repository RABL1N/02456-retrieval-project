import torch

def build_batch(rows):
    """
    @param rows: a slice of corpus.train, e.g. corpus.train[0:10] (dict of lists)
    Returns: query texts, unique document texts (all positives of the batch), (B_q, B_d) bool mask
    """
    q_texts = rows['question']

    col = {}                                  # document_id -> column index
    d_texts = []
    for dids, texts in zip(rows['document_id'], rows['text']):
        for did, txt in zip(dids, texts):
            if did not in col:
                col[did] = len(d_texts)
                d_texts.append(txt)

    pos_mask = torch.zeros(len(q_texts), len(d_texts), dtype=torch.bool)
    for i, dids in enumerate(rows['document_id']):
        for did in dids:
            pos_mask[i, col[did]] = True

    return q_texts, d_texts, pos_mask