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


def calculate_mrr(df_eval, collection, embedder, top_k=20):
    reciprocal_ranks = []

    for _, row in df_eval.iterrows():
        query_text = row['generated_question']
        eval_answer = row['used_chunk']
        doc_id_df = str(row['doc_id_df'])  # Convert to string for comparison

        # Generate query embedding
        query_embedding = embedder.embed_query(query_text)  # or local embedding
        prepared_embedding = prepare_embedding_for_chromadb(query_embedding)

        # Retrieve top-k results
        results = collection.query(
            query_embeddings=[prepared_embedding.tolist()],
            n_results=top_k,
            include=['metadatas', 'embeddings', 'documents']
        )

        # Process retrieved documents
        retrieved_docs = []
        if 'documents' in results and results['documents']:
            for idx, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][idx]
                retrieved_doc_id = results['ids'][0][idx].split('_')[-1]  # Extract only the numeric part
                retrieved_content = doc  # Assuming `doc` contains the actual document text

                # Append the necessary info for rank comparison
                retrieved_docs.append((retrieved_doc_id, metadata, retrieved_content))

        # Determine the rank of the correct document
        rank = None
        for i, (retrieved_doc_id, metadata, retrieved_content) in enumerate(retrieved_docs):
            if retrieved_doc_id == doc_id_df and eval_answer in (retrieved_content or ''):
                rank = i + 1  # 1-based index for rank
                break

        # Calculate reciprocal rank
        reciprocal_rank = 1 / rank if rank else 0.0
        reciprocal_ranks.append(reciprocal_rank)

    # Calculate Mean Reciprocal Rank (MRR)
    mrr = np.mean(reciprocal_ranks)
    print(f"Mean Reciprocal Rank (MRR): {mrr}")
    return mrr