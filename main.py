import hydra
from src.data_utils.corpus import Corpus
from src.data_utils.query import Query
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate

from src.retrievers.tf_idf import TFIDF
from src.retrievers.bm25 import BM25
from src.retrievers.dpr import DPR

from src.training.trainer import Trainer
import torch
import os

@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg):
    ### Load a dataset
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')
    ### Prepare queries
    train_queries = [Query(corpus.train['question'][idx], 
                           id=corpus.train['query_id'][idx], 
                           relevant_document_ids=corpus.relevant_ids[corpus.train['query_id'][idx]]
                           ) 
                        for idx in range(corpus.train.num_rows)
                    ]

    val_queries =   [Query(corpus.val['question'][idx], 
                         id=corpus.val['query_id'][idx], 
                         relevant_document_ids=corpus.relevant_ids[corpus.val['query_id'][idx]]
                           ) 
                        for idx in range(corpus.val.num_rows)
                    ]
    
    test_queries =  [Query(corpus.test['question'][idx], 
                           id=corpus.test['query_id'][idx], 
                           relevant_document_ids=corpus.relevant_ids[corpus.test['query_id'][idx]]
                           ) 
                        for idx in range(corpus.test.num_rows)
                    ]
    
    ### Initialize Retriever class
    retrievers = {'tf-idf': TFIDF, 'bm25': BM25, 'dpr': DPR}

    if cfg.retriever.method == 'dpr':
        run = f'ep{cfg.train.epochs}_len{cfg.train.max_length}_{cfg.retriever.similarity}'
        retriever = DPR(cfg, documents=corpus.all_documents)
        Trainer(cfg).train(retriever, corpus.train, corpus.val, save_dir=f'outputs/models/{run}')
        L = cfg.retriever.index_max_length
        retriever.encode_corpus(max_length=L)                              # no cache: always fresh weights
        retriever.save_embeddings(f'outputs/embeddings/{run}_{L}.pt', L)
    else:
        retriever = retrievers[cfg.retriever.method](documents=corpus.all_documents)
    
    ### Retrieve top-k relevant queries
    train_relevancies = retrieveQueryAndGetRelevancies(retriever, train_queries, k=cfg.top_k)
    val_relevancies   = retrieveQueryAndGetRelevancies(retriever, val_queries,   k=cfg.top_k)
    test_relevancies  = retrieveQueryAndGetRelevancies(retriever, test_queries,  k=cfg.top_k)

    ### Evaluate
    train_scores = Evaluate(train_relevancies, train_queries)
    val_scores = Evaluate(val_relevancies, val_queries)
    test_scores = Evaluate(test_relevancies, test_queries)

    print(f'Train results: {train_scores}')
    print(f'Val results: {val_scores}')
    print(f'Test results: {test_scores}')

if __name__ == '__main__':
    main()

# We can run the experiments with Hydra 
# .venv/bin/python main.py retriever=bm25
# .venv/bin/python main.py retriever=tf_idf
# .venv/bin/python main.py retriever=dpr