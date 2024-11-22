import chromadb
from tqdm import tqdm
import numpy as np
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.documents.base import Document
from langchain_chroma import Chroma

DEFAULT_K = 4


class EmbeddingVectorStorage:
    def __init__(
        self,
        method_of_embedding: Embeddings,
        collection: str,
        path_persistent: str = "../data/chroma",
    ):

        self.client = chromadb.PersistentClient(path=path_persistent)
        self.method_of_embedding = method_of_embedding
        self.collection = collection

        self.storage_of_vector = Chroma(
            client=self.client,
            collection_name=collection,
            embedding_function=method_of_embedding,
        )

    def test_heartbeat(self) -> int:
        return self.client.heartbeat()

    def reset_client(self) -> None:
        self.client.reset()

    def delete_group(self) -> None:
        self.storage_of_vector.delete_collection()

    def as_retriever(self, k: int = DEFAULT_K) -> BaseRetriever:
        return self.storage_of_vector.as_retriever(search_kwargs={"k": k})

    def include_documents(
        self,
        documents: list[Document],
        size_of_batch=41666,
        should_verbose: bool = False,
        allow_overwrite: bool = False,
    ):
        if not self.collection_is_empty() and not allow_overwrite:
            print(f"Group {self.collection} already exists in the vector storage.")
            return

        size_of_batch = min(size_of_batch, 41666)
        batches = [
            documents[i : i + size_of_batch]
            for i in range(0, len(documents), size_of_batch)
        ]

        for batch in tqdm(batches) if should_verbose else batches:
            self.storage_of_vector.add_documents(
                documents=batch, verbose=should_verbose
            )

    def search_similar(self, query: str) -> list[Document]:
        return self.storage_of_vector.similarity_search(query)

    def search_similar_w_scores(
        self, query: str, k: int = None
    ) -> list[tuple[Document, float]]:
        return self.storage_of_vector.similarity_search_with_score(query, k=DEFAULT_K)

    def does_group_exist(self) -> bool:
        return np.any(
            [group.name == self.collection for group in self.client.list_collections()]
        )

    def collection_is_empty(self) -> bool:
        return self.client.get_collection(self.collection).count() == 0

    def __repr__(self) -> str:
        return (
            f"VectorStorage(method_of_embedding={self.method_of_embedding.__class__.__name__}, "
            f"group={self.collection})"
        )
