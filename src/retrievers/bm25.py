from scipy.sparse import lil_matrix
from src.retrievers.tf_idf import TFIDF
from src.data_utils.document import Document
import numpy as np

class BM25(TFIDF):
    def __init__(self, documents: list[dict] = None, index_path: str = None, k1: float = 1.5, b: float = 0.75) -> None:
        super(BM25, self).__init__(documents, index_path)
        self.k1 = k1
        self.b = b
        # Added this line, similiar of how it's done in tf_idf.py, for initalizing BM25 matrix
        if type(self) == BM25:
            self.bm25_matrix = self.GetDocumentsBM25Vectors()



    def GetDocumentLengths(self):
        document_lengths = {}
        average_document_length = 0
        for document in self.index.GetDocuments():
            length = len(self.PreprocessText(document.GetText()))
            document_lengths[document] = length
            average_document_length += length
        return document_lengths, average_document_length/len(self.index.GetDocuments())
    
    # TODO: Add methods for computing self.bm25_matrix
    
    # Added inverse document frequencies by copying the function from tf_idf.py and adjusting to the formula written on the project description
    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies, now adjusted to the BM25 formula
        """

        N = len(self.index.GetDocuments()) # number of documents

        doc_frequency = self.GetDocumentFrequencies() # n_qi

        idf_bm25 = {term: np.log((N + 1.0) / (df + 0.5)) for term, df in doc_frequency.items()} # inverse document frequency dict

        return idf_bm25


    #Added GetDocumentsBM25Vectors based on the added works in tf_idf (same matrix structure but replacing the values for the bm25 adjusted tf and idf)

    def GetDocumentsBM25Vectors(self):
        """
        Returns a n_documents x n_terms matrix whose element (i, j) is the tf-idf score of the j'th term in the i'th document, but adjusted to the formula
        """
        # Create a mapping from terms to indices
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}

        # Initialize a sparse matrix
        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)
        bm25_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)

        #Values needed to the formula
        document_len, average_document_len = self.GetDocumentLengths()
        documents = self.index.GetDocuments()

        for doc_idx, document in enumerate(documents):
            term_count = self.GetDocumentTermCounts(document)
            doc_len = document_len[document]
            
            for term, counts in term_count.items():
                tf_25= (counts * (self.k1+1.0))/ (counts+ self.k1*(1-self.b+ self.b*(doc_len/average_document_len)))
                bm25_matrix[doc_idx, self.term_to_index[term]] = tf_25 * self.idf[term]
        
        return bm25_matrix.tocsr()   # Convert to CSR format for efficient row slicing



    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document
        """
        query_vector = self.QueryToVector(queries)
        scores = query_vector.dot(self.bm25_matrix.T).toarray()
        return scores