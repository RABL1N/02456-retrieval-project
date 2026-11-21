from src.retrievers.models.encoder import TransformerEncoder

class DPR:
    def __init__(self, cfg):
        super().__init__() # any nn.Module subclass needs this
        self.cfg = cfg
        self.__load_models()
    
    def __load_models(self):
        # n_heads, n_layers, emb_in, hidden_dim, feed_forward_dim

        self.query_encoder = TransformerEncoder(
            pooling_type=self.cfg.model.query_encoder.pooling_type
        )
        self.doc_encoder = TransformerEncoder(
            pooling_type=self.cfg.model.doc_encoder.pooling_type
        )

        
        query_weights_path = self.cfg.model.get('query_encoder_weights_path', None)
        if self.cfg.model.get('query_encoder_weights_path', None):
            self.query_encoder.load_state_dict(query_weights_path)

        document_weights_path = self.cfg.model.get('document_encoder_weights_path', None)
        if self.cfg.model.get('query_encoder_weights_path', None):
            self.doc_encoder.load_state_dict(document_weights_path)


    # def query_document(self, queries, documents):
    #     """
    #     Returns similarity matrix of size n_documents * n_queries between 
    #     the document embeddings and a query embeddings
    #     """

    #     input_query = self.query_encoder.tokenizer(
    #         queries,
    #         return_tensors="pt")
    
    #     input_doc = self.doc_encoder.tokenizer(
    #         documents,
    #         return_tensors="pt")
    
    #     output_query = self.query_encoder.model(**input_query)
    #     output_doc = self.doc_encoder.model(**input_doc)
    
    #     # Pooling
    #     query_vectors = output_query.pooler_output
    #     doc_vectors = output_doc.pooler_output
    
    #     dot_product = query_vectors @ doc_vectors.T
    
    #     return dot_product


    def encode_queries(self, queries):          # list[str] -> (B, D)
        inputs = self.query_encoder.tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
        return self.query_encoder.model(**inputs).pooler_output

    def encode_documents(self, documents):      # list[str] -> (B, D)
        inputs = self.doc_encoder.tokenizer(documents, padding=True, truncation=True, return_tensors="pt")
        return self.doc_encoder.model(**inputs).pooler_output

    def query_document(self, queries, documents):
        return self.encode_queries(queries) @ self.encode_documents(documents).T
