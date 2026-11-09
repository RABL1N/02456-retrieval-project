from src.data_utils.document import Document
from src.retrievers.base_retriever import Retriever
import numpy as np
from scipy.sparse import lil_matrix


class TFIDF(Retriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        super(TFIDF, self).__init__(documents, index_path)
        self.corpus_vocabulary = self.GetCorpusVocabulary()
        self.idf = self.GetInverseDocumentFrequencies()

        # Only run if we are in TFIDF class
        if type(self) == TFIDF:
            self.tfidf_matrix = self.GetDocumentsTFIDFVectors()


    def PreprocessText(self, text: str) -> list[str]:
        """
        Remove unwanted characters from text and lowercase
        """
        
        return text.lower().split()
        
        # raise NotImplementedError()


    def GetQueryVocabulary(self, query: str) -> list[str]:
        """
        Returns all unique words in a query
        """

        return list(set(self.PreprocessText(query)))

        #raise NotImplementedError()


    def GetDocumentVocabulary(self, document: Document) -> list[str]:
        """
        Returns all unique words in a document
        """

        return set(self.PreprocessText(document.GetText()))

        # raise NotImplementedError()
    

    def GetCorpusVocabulary(self) -> set[str]:
        """
        Returns all unique words in ALL documents.
        """
        # Use map function for more efficient processing
        processed_texts = map(lambda doc: set(self.PreprocessText(doc.GetText())), self.index.GetDocuments())
        corpus_vocabulary = set().union(*processed_texts)
        return corpus_vocabulary


    def GetDocumentFrequencies(self) -> dict[str, int]:
        """
        Returns a dict with terms and counts of how many documents the term occurs in (counted once per document).
        """

        df = {} # document frequency of all terms in corpus vocab

        for doc in self.index.GetDocuments():

            for term in self.GetDocumentVocabulary(doc):
                
                df[term] = df.get(term, 0) + 1

        return df

        # raise NotImplementedError()
       

    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies.
        """

        N = len(self.index.GetDocuments()) # number of documents

        doc_frequency = self.GetDocumentFrequencies() # n_qi

        idf = {term: np.log(N / df) for term, df in doc_frequency.items()} # inverse document frequency dict

        return idf

        # raise NotImplementedError()


    def GetDocumentTermCounts(self, document: Document):
        """
        Returns a dict on the form {term: num_occurences} for an input Document.
        """

        terms = self.GetDocumentVocabulary(document)
        counts = []

        for term in terms:
            count = document.GetText().count(term)
            counts.append(count)

        term_counts = dict(zip(terms, counts))

        return term_counts        

        # raise NotImplementedError()


    def GetDocumentsTFIDFVectors(self):
        """
        Returns a n_documents x n_terms matrix whose element (i, j) is the td-idf score of the j'th term in the i'th document.
        """
        # Create a mapping from terms to indices
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}

        # Initialize a sparse matrix
        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)
        tfidf_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)

        # TODO: Add tf-idf code

        documents = self.index.GetDocuments()

        for doc_idx, document in enumerate(documents):
            term_count = self.GetDocumentTermCounts(document)
            max_term_count = max(term_count.values())
            
            for term, counts in term_count.items():
                tfidf_matrix[doc_idx, self.term_to_index[term]] = (counts / max_term_count) * self.idf[term]

        
        return tfidf_matrix.tocsr()   # Convert to CSR format for efficient row slicing
    
    def QueryToVector(self, queries: list[str]):
        """ Converts a query or batch of queries into a sparse vector matching the vocab size with counts of how many times the term occurs.
        
        Args:
            query (str): The query to convert
        
        Returns:
            csr_matrix: A sparse vector representation of the query
        """
        for q in queries:
            query_vocabulary = self.GetQueryVocabulary()
        query_vector = lil_matrix((len(queries), len(self.corpus_vocabulary)), dtype=np.float32)
        

        # for query_id, query in enumerate(queries):
        #     term_count = self.GetDocumentTermCounts(query)
        #     for term, counts in term_count.items():
        #         query_vector[query_id, self.term_to_index[term]] = counts

        return query_vector.tocsr()

    #@time_func
    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document

            num_queries x num_documents where (i, j) = ∑_i TF-IDF(q_i, D_j; D) 
        
        """
        # Convert the queries to a vector
        query_vectors = self.QueryToVector(queries) # n_queries x vocab size
        scores = query_vectors.dot(self.tfidf_matrix.T).toarray() # n_queries x n_documents
        return scores