"""Document Database and Retrieval Module."""
import json
import time
import os
from typing import Optional, List, Dict
import sqlite3
import numpy as np
import pickle as pkl

from rank_bm25 import BM25Okapi
from transformers import RobertaTokenizer
from sentence_transformers import SentenceTransformer

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
MAX_LENGTH = 256

class DocDB(object):
    """SQLite-backed Document Storage.

    Implements get_doc_text(doc_id).

    Attributes:
        db_path (str): Path to the SQLite database file.
        connection (sqlite3.Connection): SQLite connection object.
        add_n (int): Counter for the number of new documents added to the cache.
    """    
    def __init__(self, db_path: Optional[str] = None, data_path: Optional[str] = None) -> None:
        """Initialize the DocDB instance.

        Connects to the SQLite database at `db_path`. If the database is empty, it builds the database
        from the provided `data_path`.

        Args:
            db_path (Optional[str], optional): Path to the SQLite database file. Defaults to None.
            data_path (Optional[str], optional): Path to the raw data file for building the database.
                Required if the database does not exist or is empty. Defaults to None.

        Raises:
            AssertionError: If `data_path` is not provided when the database is empty.
        """
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)

        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        
        if len(cursor.fetchall())==0:
            assert data_path is not None, f"{self.db_path} is empty. Specify `data_path` in order to create a DB."
            print (f"{self.db_path} is empty. start building DB from {data_path}...")
            self.build_db(self.db_path, data_path)

    def __enter__(self) -> 'DocDB':
        """Enter the runtime context related to this object.

        Returns:
            DocDB: The DocDB instance itself.
        """
        return self
    def __exit__(self, *args) -> None:
        """Exit the runtime context and close the database connection."""
        self.close()

    def path(self) -> str:
        """Return the path to the file that backs this database.

        Returns:
            str: Path to the SQLite database file.        
        """
        return self.path

    def close(self) -> None:
        """Close the connection to the database."""
        self.connection.close()

    def build_db(self, db_path: str, data_path: str) -> None:
        """Build the SQLite database from raw JSON data.

        This method reads raw data from `data_path`, processes it using a tokenizer, and inserts
        the documents into the SQLite database.

        Args:
            db_path (str): Path to the SQLite database file.
            data_path (str): Path to the raw data file (JSON lines format).

        Raises:
            AssertionError: If a sentence in the text is empty.
        """
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        
        titles = set()
        output_lines = []
        tot = 0
        start_time = time.time()
        c = self.connection.cursor()
        c.execute("CREATE TABLE documents (title PRIMARY KEY, text);")

        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                title = dp["title"]
                text = dp["text"]
                if title in titles:
                    continue
                titles.add(title)
                if type(text)==str:
                    text = [text]
                passages = [[]]
                for sent_idx, sent in enumerate(text):
                    assert len(sent.strip())>0
                    tokens = tokenizer(sent)["input_ids"]
                    max_length = MAX_LENGTH - len(passages[-1])
                    if len(tokens) <= max_length:
                        passages[-1].extend(tokens)
                    else:
                        passages[-1].extend(tokens[:max_length])
                        offset = max_length
                        while offset < len(tokens):
                            passages.append(tokens[offset:offset+MAX_LENGTH])
                            offset += MAX_LENGTH
                
                psgs = [tokenizer.decode(tokens) for tokens in passages if np.sum([t not in [0, 2] for t in tokens])>0]
                text = SPECIAL_SEPARATOR.join(psgs)
                output_lines.append((title, text))
                tot += 1

                if len(output_lines) == 1000000:
                    c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
                    output_lines = []
                    print ("Finish saving %dM documents (%dmin)" % (tot / 1000000, (time.time()-start_time)/60))

        if len(output_lines) > 0:
            c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
            print ("Finish saving %dM documents (%dmin)" % (tot / 1000000, (time.time()-start_time)/60))

        self.connection.commit()
        self.connection.close()

    def get_text_from_title(self, title: str) -> List[Dict[str, str]]:
        """Fetch the raw text of the doc for 'doc_id'.

        Args:
            title (str): The title of the document to fetch.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing the title and text passages.

        Raises:
            AssertionError: If the title does not exist in the database or has no valid passages.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
        results = cursor.fetchall()
        results = [r for r in results]
        cursor.close()
        assert results is not None and len(results)==1, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        results = [{"title": title, "text": para} for para in results[0][0].split(SPECIAL_SEPARATOR)]
        assert len(results)>0, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        return results

class Retrieval(object):
    """Document Retrieval Class.

    Attributes:
        db (DocDB): Instance of the DocDB class for accessing documents.
        cache_path (str): Path to the JSON cache file for storing retrieval results.
        embed_cache_path (str): Path to the pickle cache file for storing embeddings.
        retrieval_type (str): Type of retrieval method to use ('bm25' or transformer-based).
        batch_size (Optional[int]): Batch size for embedding computations. Required for transformer-based retrieval.
        encoder (Optional[SentenceTransformer]): Sentence transformer model for embedding-based retrieval.
        cache (Dict[str, Any]): Cache dictionary for retrieval results.
        embed_cache (Dict[str, Any]): Cache dictionary for embeddings.
        add_n (int): Counter for the number of new retrieval entries added to the cache.
        add_n_embed (int): Counter for the number of new embeddings added to the cache.
    """
    def __init__(
        self,
        db: DocDB,
        cache_path: str,
        embed_cache_path: str,
        retrieval_type: str = "gtr-t5-large",
        batch_size: Optional[int] = None
    ) -> None:
        """Initialize the Retrieval instance.

        Args:
            db (DocDB): Instance of the DocDB class for accessing documents.
            cache_path (str): Path to the JSON cache file for storing retrieval results.
            embed_cache_path (str): Path to the pickle cache file for storing embeddings.
            retrieval_type (str, optional): Type of retrieval method to use ('bm25' or transformer-based).
                Defaults to "bm25".
            batch_size (Optional[int], optional): Batch size for embedding computations. Required for
                transformer-based retrieval. Defaults to None.

        Raises:
            AssertionError: If `retrieval_type` is not 'bm25' or does not start with 'gtr-'.
        """
        self.db = db
        self.cache_path = cache_path
        self.embed_cache_path = embed_cache_path
        self.retrieval_type = retrieval_type
        self.batch_size = batch_size
        assert retrieval_type=="bm25" or retrieval_type.startswith("gtr-")
        
        self.encoder = None
        self.load_cache()
        self.add_n = 0
        self.add_n_embed = 0

    def load_encoder(self) -> None:
        """Load the sentence transformer encoder for embedding-based retrieval.

        Raises:
            ValueError: If `batch_size` is not set for transformer-based retrieval.
        """
        encoder = SentenceTransformer("sentence-transformers/" + self.retrieval_type)
        encoder = encoder.cuda()
        encoder = encoder.eval()
        self.encoder = encoder
        assert self.batch_size is not None
    
    def load_cache(self) -> None:
        """Load retrieval and embedding caches from the specified cache files."""
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
        if os.path.exists(self.embed_cache_path):
            with open(self.embed_cache_path, "rb") as f:
                self.embed_cache = pkl.load(f)
        else:
            self.embed_cache = {}
    
    def save_cache(self) -> None:
        """Save retrieval and embedding caches to the specified cache files."""
        if self.add_n > 0:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r") as f:
                    new_cache = json.load(f)
                self.cache.update(new_cache)
            
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)
        
        if self.add_n_embed > 0:
            if os.path.exists(self.embed_cache_path):
                with open(self.embed_cache_path, "rb") as f:
                    new_cache = pkl.load(f)
                self.embed_cache.update(new_cache)
            
            with open(self.embed_cache_path, "wb") as f:
                pkl.dump(self.embed_cache, f)

    def get_bm25_passages(
        self,
        topic: str,
        query: str,
        passages: List[Dict[str, str]],
        k: int
    ) -> List[Dict[str, str]]:
        """Retrieve top-k passages using BM25.

        Args:
            topic (str): The topic associated with the query.
            query (str): The query string.
            passages (List[Dict[str, str]]): List of passages associated with the topic.
            k (int): Number of top passages to retrieve.

        Returns:
            List[Dict[str, str]]: List of top-k retrieved passages.
        """
        if topic in self.embed_cache:
            bm25 = self.embed_cache[topic]
        else:
            bm25 = BM25Okapi([psg["text"].replace("<s>", "").replace("</s>", "").split() for psg in passages])
            self.embed_cache[topic] = bm25
            self.add_n_embed += 1
        scores = bm25.get_scores(query.split())
        indices = np.argsort(-scores)[:k]
        return [passages[i] for i in indices]

    def get_gtr_passages(
        self,
        topic: str,
        retrieval_query: str,
        passages: List[Dict[str, str]],
        k: int
    ) -> List[Dict[str, str]]:
        """Retrieve top-k passages using transformer-based retrieval (e.g., GTR).

        Args:
            topic (str): The topic associated with the query.
            retrieval_query (str): The query string.
            passages (List[Dict[str, str]]): List of passages associated with the topic.
            k (int): Number of top passages to retrieve.

        Returns:
            List[Dict[str, str]]: List of top-k retrieved passages.
        """
        if self.encoder is None:
            self.load_encoder()
        if topic in self.embed_cache:
            passage_vectors = self.embed_cache[topic]
        else:
            inputs = [psg["title"] + " " + psg["text"].replace("<s>", "").replace("</s>", "") for psg in passages]
            passage_vectors = self.encoder.encode(inputs, batch_size=self.batch_size, device=self.encoder.device)
            self.embed_cache[topic] = passage_vectors
            self.add_n_embed += 1
        query_vectors = self.encoder.encode([retrieval_query], 
                                            batch_size=self.batch_size,
                                            device=self.encoder.device)[0]
        scores = np.inner(query_vectors, passage_vectors)
        indices = np.argsort(-scores)[:k]
        return [passages[i] for i in indices]

    def get_passages(
        self,
        topic: str,
        question: str,
        k: int
    ) -> List[Dict[str, str]]:
        """Retrieve top-k passages based on the topic and question using the specified retrieval method.

        Args:
            topic (str): The topic associated with the query.
            question (str): The question string.
            k (int): Number of top passages to retrieve.

        Returns:
            List[Dict[str, str]]: List of top-k retrieved passages.
        """
        retrieval_query = topic + " " + question.strip()
        cache_key = topic + "#" + retrieval_query
        
        if cache_key not in self.cache:
            passages = self.db.get_text_from_title(topic)
            if self.retrieval_type=="bm25":
                self.cache[cache_key] = self.get_bm25_passages(topic, retrieval_query, passages, k)
            else:
                self.cache[cache_key] = self.get_gtr_passages(topic, retrieval_query, passages, k)
            assert len(self.cache[cache_key]) in [k, len(passages)]
            self.add_n += 1
        
            
        return self.cache[cache_key]