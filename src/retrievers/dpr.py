import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_utils.dataset import DocumentDataset

from src.retrievers.models.encoder import TransformerEncoder

from src.retrievers.base_retriever import Retriever

from src.retrievers.utils import resolve_device

class DPR(nn.Module, Retriever):
    def __init__(self, cfg, documents=None, index_path=None):
        nn.Module.__init__(self)                              # must come first
        Retriever.__init__(self, documents, index_path)       # sets self.index (DocumentDataset)
        self.cfg = cfg
        self.__load_models()
    
    def __load_models(self):
        # n_heads, n_layers, emb_in, hidden_dim, feed_forward_dim

        self.device = resolve_device(self.cfg.device)

        self.query_encoder = TransformerEncoder(
            pooling_type=self.cfg.model.query_encoder.pooling_type
        )
        self.query_encoder.to(self.device)

        self.doc_encoder = TransformerEncoder(
            pooling_type=self.cfg.model.doc_encoder.pooling_type
        )
        self.doc_encoder.to(self.device)

        query_weights_path = self.cfg.model.get('query_encoder_weights_path', None)
        if query_weights_path:
            self.query_encoder.load_state_dict(torch.load(query_weights_path, map_location=self.device))

        document_weights_path = self.cfg.model.get('document_encoder_weights_path', None)
        if document_weights_path:                     # was checking the query path
            self.doc_encoder.load_state_dict(torch.load(document_weights_path, map_location=self.device))


    def _normalize(self, x):
        """Unit-length embeddings when using cosine similarity; unchanged for dot product."""
        return F.normalize(x, dim=-1) if self.cfg.retriever.similarity == 'cosine' else x


    def encode_queries(self, queries):                              # list[str] -> (B, D)
        inputs = self.query_encoder.tokenizer(queries, padding=True, truncation=True, max_length=64, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self._normalize(self.query_encoder(inputs))


    def encode_documents(self, documents, max_length=256):          # list[str] -> (B, D)
        inputs = self.doc_encoder.tokenizer(documents, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self._normalize(self.doc_encoder(inputs))
    

    def query_document(self, queries, documents):
        return self.encode_queries(queries) @ self.encode_documents(documents).T


    # @torch.no_grad()
    # def encode_corpus(self, batch_size=64, max_length=512):
    #     """Embed every document in self.index. Call after training / whenever weights change."""
    #     self.eval()
    #     texts = [d.GetText() for d in self.index.GetDocuments()]
    #     embs = [self.encode_documents(texts[s:s + batch_size], max_length=max_length).cpu()
    #             for s in range(0, len(texts), batch_size)]
    #     self.doc_embeddings = torch.cat(embs)                 # (N_docs, D)


    @torch.no_grad()
    def encode_corpus(self, batch_size=64, max_length=512, cache_path=None):
        """
        Embed every document in self.index.
        @param cache_path: if given, load embeddings from it when it exists and matches
                           the current settings; otherwise encode and save to it.
        """
        if cache_path and os.path.exists(cache_path):
            try:
                self.load_embeddings(cache_path, max_length)
                print(f'Loaded document embeddings from {cache_path}')
                return
            except AssertionError as e:
                print(f'Cache does not match ({e}), re-encoding')

        self.eval()
        texts = [d.GetText() for d in self.index.GetDocuments()]
        embs = [self.encode_documents(texts[s:s + batch_size], max_length=max_length).cpu()
                for s in range(0, len(texts), batch_size)]
        self.doc_embeddings = torch.cat(embs)                      # (N_docs, D)

        if cache_path:
            self.save_embeddings(cache_path, max_length)


    def save_embeddings(self, path, max_length):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'embeddings': self.doc_embeddings,
            'doc_ids': [d.GetId() for d in self.index.GetDocuments()],
            'pooling_type': self.doc_encoder.pooling_type,
            'max_length': max_length,
        }, path)


    def load_embeddings(self, path, max_length=None):
        saved = torch.load(path)
        assert saved['pooling_type'] == self.doc_encoder.pooling_type, \
            f"pooling {saved['pooling_type']} != {self.doc_encoder.pooling_type}"
        assert max_length is None or saved['max_length'] == max_length, \
            f"max_length {saved['max_length']} != {max_length}"
        assert saved['doc_ids'] == [d.GetId() for d in self.index.GetDocuments()], \
            'document order differs'
        self.doc_embeddings = self._normalize(saved['embeddings'])


    @torch.no_grad()
    def CalculateScores(self, queries: list[str]):
        """(n_queries, N_docs) score matrix, same contract as TF-IDF/BM25."""
        self.eval()
        return self.encode_queries(queries) @ self.doc_embeddings.to(self.device).T


    @torch.no_grad()
    def Lookup(self, queries: list[str], k: int, batch_size=256):
        """Override base Lookup: topk instead of a Python sort over all documents."""
        documents = self.index.GetDocuments()
        results = []
        for s in range(0, len(queries), batch_size):
            scores = self.CalculateScores(queries[s:s + batch_size])
            top = scores.topk(min(k, scores.size(1)), dim=1).indices.cpu().tolist()
            results.extend([[documents[j] for j in row] for row in top])
        return results