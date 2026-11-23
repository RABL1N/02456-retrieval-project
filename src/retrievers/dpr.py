import torch
import torch.nn as nn

from src.data_utils.dataset import DocumentDataset

from src.retrievers.models.encoder import TransformerEncoder

from src.retrievers.base_retriever import Retriever

from src.retrievers.utils import resolve_device

class DPR(nn.Module, Retriever):
    def __init__(self, cfg, documents=None, index_path=None):
        super().__init__() # any nn.Module subclass needs this
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
        if self.cfg.model.get('query_encoder_weights_path', None):
            self.query_encoder.load_state_dict(query_weights_path)

        document_weights_path = self.cfg.model.get('document_encoder_weights_path', None)
        if self.cfg.model.get('query_encoder_weights_path', None):
            self.doc_encoder.load_state_dict(document_weights_path)


    def encode_queries(self, queries):          # list[str] -> (B, D)
        inputs = self.query_encoder.tokenizer(queries, padding=True, truncation=True, max_length=64, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.query_encoder.model(**inputs).pooler_output


    def encode_documents(self, documents, max_length=256):      # list[str] -> (B, D)
        inputs = self.doc_encoder.tokenizer(documents, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.doc_encoder.model(**inputs).pooler_output
    

    def query_document(self, queries, documents):
        return self.encode_queries(queries) @ self.encode_documents(documents).T



    @torch.no_grad()
    def encode_corpus(self, batch_size=64, max_length=512):
        """Embed every document in self.index. Call after training / whenever weights change."""
        self.eval()
        texts = [d.GetText() for d in self.index.GetDocuments()]
        embs = [self.encode_documents(texts[s:s + batch_size], max_length=max_length).cpu()
                for s in range(0, len(texts), batch_size)]
        self.doc_embeddings = torch.cat(embs)                 # (N_docs, D)

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