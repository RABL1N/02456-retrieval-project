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

    