import torch
from src.training.loss import DPRLoss
from src.training.batching import build_batch

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        t = self.cfg.retriever.temperature if self.cfg.retriever.similarity == 'cosine' else 1.0
        self.criterion = DPRLoss(temperature=t)

    def _batch_loss(self, dpr, rows):
        q_texts, d_texts, pos_mask = build_batch(rows, max_pos=self.cfg.train.max_positives)
        q = dpr.encode_queries(q_texts)
        d = dpr.encode_documents(d_texts, max_length=self.cfg.train.max_length)
        return self.criterion(q, d, pos_mask)

    @torch.no_grad()
    def evaluate_loss(self, dpr, ds):
        """Average in-batch loss on a split (e.g. corpus.val), cheap way to monitor overfitting."""
        dpr.eval()
        B = self.cfg.train.batch_size
        losses = [self._batch_loss(dpr, ds[s:s + B]).item() for s in range(0, len(ds), B)]
        return sum(losses) / len(losses)

    def train(self, dpr, train_ds, val_ds=None):
        """
        @param dpr: DPR model (nn.Module containing both encoders)
        @param train_ds: corpus.train
        @param val_ds: corpus.val, optional, for validation loss after each epoch
        """
        if self.cfg.train.gradient_checkpointing:
            dpr.query_encoder.model.gradient_checkpointing_enable()
            dpr.doc_encoder.model.gradient_checkpointing_enable()

        opt = torch.optim.AdamW(dpr.parameters(), lr=self.cfg.train.lr)
        B = self.cfg.train.batch_size

        for epoch in range(self.cfg.train.epochs):
            dpr.train()                                         # dropout on
            order = torch.randperm(len(train_ds)).tolist()      # new shuffle each epoch
            running = 0.0
            for step, s in enumerate(range(0, len(order), B)):
                loss = self._batch_loss(dpr, train_ds[order[s:s + B]])
                opt.zero_grad()
                loss.backward()
                opt.step()
                if dpr.device.type == 'mps':
                    torch.mps.empty_cache()     # release cached blocks so memory doesn't creep up


                running += loss.item()
                if (step + 1) % self.cfg.train.log_every == 0:
                    print(f'epoch {epoch} step {step + 1} train loss {running / self.cfg.train.log_every:.4f}')
                    running = 0.0

            if val_ds is not None:
                print(f'epoch {epoch} val loss {self.evaluate_loss(dpr, val_ds):.4f}')

        dpr.eval()