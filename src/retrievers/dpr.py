import torch.nn as nn

from src.retrievers.models.encoder import TransformerEncoder

from src.retrievers.utils import resolve_device

class DPR(nn.Module):
    def __init__(self, cfg):
        super().__init__() # any nn.Module subclass needs this
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
        
        # inputs = self.query_encoder.tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
        # return self.query_encoder.model(**inputs).pooler_output


    def encode_documents(self, documents):      # list[str] -> (B, D)
        inputs = self.doc_encoder.tokenizer(documents, padding=True, truncation=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.doc_encoder.model(**inputs).pooler_output

        # inputs = self.doc_encoder.tokenizer(documents, padding=True, truncation=True, return_tensors="pt")
        # return self.doc_encoder.model(**inputs).pooler_output


    def query_document(self, queries, documents):
        return self.encode_queries(queries) @ self.encode_documents(documents).T
