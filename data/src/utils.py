import numpy as np
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity

# Flatten, pad/truncate, and convert each embedding to a consistent 1D np.float32 array
def prepare_embedding_for_chromadb(embedding):
    # Flatten the embedding if it's nested
    flat_embedding = [float(val) for sublist in embedding for val in sublist] if isinstance(embedding[0], (
    list, np.ndarray)) else embedding

    # Ensure the embedding is exactly 2048 dimensions
    if len(flat_embedding) < 2048:
        flat_embedding.extend([0.0] * (2048 - len(flat_embedding)))  # Pad with zeros if too short
    elif len(flat_embedding) > 2048:
        flat_embedding = flat_embedding[:2048]  # Truncate if too long

    # Convert to np.float32
    return np.array(flat_embedding, dtype=np.float32)


def split_text(documents: list[Document], text_splitter):
    chunks = text_splitter.split_documents(documents)

    return chunks


def prepare_embedding_for_comparison(embedding, target_dim=2048):
    # Flatten the embedding if it's nested
    flat_embedding = [float(val) for sublist in embedding for val in sublist] if isinstance(embedding[0], (
    list, np.ndarray)) else embedding

    # Ensure the embedding is exactly target_dim dimensions
    if len(flat_embedding) < target_dim:
        flat_embedding.extend([0.0] * (target_dim - len(flat_embedding)))  # Pad with zeros if too short
    elif len(flat_embedding) > target_dim:
        flat_embedding = flat_embedding[:target_dim]  # Truncate if too long

    # Convert to np.float32
    return np.array(flat_embedding, dtype=np.float32)

# Cosine similarity function
def cosine_similarity_score(retrieved_embeddings, relevant_embedding):
    # Calculate cosine similarity for each retrieved document with the relevant chunk
    similarities = cosine_similarity(retrieved_embeddings, [relevant_embedding])
    return similarities