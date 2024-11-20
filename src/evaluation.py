import os
import pickle
import nest_asyncio
import asyncio
from pathlib import Path
import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm
from pandarallel import pandarallel
from datasets import Dataset
import numpy as np
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_entity_recall,
    answer_similarity,
    answer_correctness,
)
from ragas.llms import LangchainLLMWrapper
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import CACHE_DIR

nest_asyncio.apply()

class RAGEvaluation:
    def __init__(self, name, rag_chain, llm_model, embeddings, local_llm=None):
        self.name = name
        self.rag_chain = rag_chain
        self.llm_model = llm_model
        self.embeddings = embeddings
        self.local_llm = local_llm
        self.clean_dataset = None
        self.eval_test = None
        self.dataset = None
        self.evaluation_result = None

    @staticmethod
    def compute_similarity(text_1, text_2):
        return fuzz.partial_ratio(text_1, text_2)

    @staticmethod
    def find_most_similar(clean_df, eval_input_chunk):
        similarity_score = 0
        similarity_score_index = None
        for idx, doc in clean_df.iterrows():
            score = fuzz.partial_ratio(eval_input_chunk, doc['content'])
            if score > similarity_score:
                similarity_score = score
                similarity_score_index = doc['id']
        return similarity_score, similarity_score_index

    @staticmethod
    def apply_find_most_similar(row, clean_df):
        score, score_index = RAGEvaluation.find_most_similar(clean_df, row['relevant_text'])
        return pd.Series([score, score_index])

    @staticmethod
    def save_to_cache(data, cache_file):
        os.makedirs(Path(cache_file).parent, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data cached at: {cache_file}")

    @staticmethod
    def load_from_cache(cache_file):
        with open(cache_file, 'rb') as f:
            print(f"Loading cached data from: {cache_file}")
            return pickle.load(f)

    def get_dynamic_filename(self, base_name, step):
        embedding_model_name = getattr(self.embeddings, "model_name", "unknown_embedding_model")
        prefix = self.local_llm + "_" if self.local_llm else ""

        if step == "preprocess":
            return CACHE_DIR / f"{base_name}.pkl"
        elif step == "dataset":
            return CACHE_DIR / f"{prefix}{embedding_model_name}_dataset.pkl"
        elif step == "evaluation":
            global_llm_name = getattr(self.llm_model, "model_name", "unknown_global_llm")
            return CACHE_DIR / f"{prefix}{embedding_model_name}_{global_llm_name}_eval_result.pkl"
        else:
            raise ValueError(f"Unknown step: {step}")

    def preprocess(self, clean_file, eval_file):
        cache_file = self.get_dynamic_filename("preprocessed", "preprocess")
        cache_path = Path(cache_file)

        if cache_path.exists():
            self.eval_test = self.load_from_cache(cache_file)
            return

        print("Preprocessing data...")
        self.clean_dataset = pd.read_parquet(clean_file)
        self.eval_test = pd.read_csv(eval_file, sep=';')

        pandarallel.initialize(progress_bar=True)
        self.eval_test[['top_score', 'top_score_id']] = self.eval_test.parallel_apply(
            RAGEvaluation.apply_find_most_similar, args=(self.clean_dataset,), axis=1
        )
        self.eval_test = self.eval_test.drop_duplicates().copy()
        self.eval_test = self.eval_test.rename(columns={'answer': 'ground_truth'})

        self.save_to_cache(self.eval_test, cache_file)

    def prepare_dataset(self):
        cache_file = self.get_dynamic_filename("dataset", "dataset")
        cache_path = Path(cache_file)

        if cache_path.exists():
            self.dataset = self.load_from_cache(cache_file)
            return

        print("Preparing evaluation dataset...")
        questions = self.eval_test['question'].tolist()
        ground_truth = self.eval_test['ground_truth'].tolist()
        ground_truth_nested = [[item] for item in ground_truth]

        data = {
            "user_input": [],
            "retrieved_contexts": [],
            "reference_contexts": ground_truth_nested,
            "response": [],
            "reference": ground_truth,
        }

        for query in tqdm(questions, desc="Preprocessing queries"):
            result_chain = self.rag_chain.invoke(query)

            # Extract plain text content for retrieved_contexts
            retrieved_contexts = [doc.page_content for doc in result_chain["context"]]

            data["user_input"].append(query)
            data["retrieved_contexts"].append(retrieved_contexts)  # List of strings
            data["response"].append(result_chain['answer'])

        # Convert to Dataset without modifying the data types
        self.dataset = Dataset.from_dict(data)
        self.save_to_cache(self.dataset, cache_file)

    def evaluate(self, clean_file_path, eval_file_path, vector_db, ks=[2], k_mrr=2):
        cache_file = self.get_dynamic_filename("evaluation_result", "evaluation")
        cache_path = Path(cache_file)

        if cache_path.exists():
            self.evaluation_result = self.load_from_cache(cache_file)
            if isinstance(self.evaluation_result, pd.DataFrame):
                return self.evaluation_result
            elif hasattr(self.evaluation_result, 'to_pandas'):
                return self.evaluation_result.to_pandas()
            else:
                raise TypeError("Cached evaluation result is not in a compatible format.")

        self.preprocess(clean_file_path, eval_file_path)
        self.prepare_dataset()

        print("Evaluating the RAG system...")
        run_config = RunConfig(timeout=3600)
        evaluator_llm = LangchainLLMWrapper(self.llm_model)

        ragas_metrics = [
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
            context_entity_recall,
            answer_similarity,
            answer_correctness,
        ]

        self.evaluation_result = evaluate(
            dataset=self.dataset,
            metrics=ragas_metrics,
            llm=evaluator_llm,
            embeddings=self.embeddings,
            raise_exceptions=True,
            show_progress=True,
            run_config=run_config,
        )

        result = self.evaluation_result.to_pandas()

        print("Calculating non-LLM-based metrics...")
        mrr = self.compute_mrr(vector_db, k=k_mrr)
        self.compute_precision_at_k(vector_db, ks=[2])
        self.compute_recall_at_k(vector_db, ks=[2])

        result['MRR'] = mrr
        result[f'precision@2'] = self.eval_test[f'precision@2']
        result[f'recall@2'] = self.eval_test[f'recall@2']

        self.save_to_cache(result, cache_file)

        print("Evaluation complete.")
        return result

    def compute_mrr(self, vector_db, k=2):
        rrs = []
        for _, row in tqdm(self.eval_test.iterrows(), desc="Computing MRR", total=len(self.eval_test)):
            query = row['question']
            retrieved_docs = vector_db.search_similar_w_scores(query, k=k)
            retrieved_doc_ids = [doc[0].metadata['origin_doc_id'] for doc in retrieved_docs]
            ground_truth_id = row['top_score_id']

            try:
                index = retrieved_doc_ids.index(ground_truth_id)
                rr = 1 / (index + 1)
            except ValueError:
                rr = 0
            rrs.append(rr)

        self.eval_test['rr'] = rrs
        mrr = np.mean(rrs)
        print(f"MRR: {mrr}")
        return mrr

    def compute_precision_at_k(self, vector_db, ks=[2]):
        k = ks[0]
        self.eval_test[f'precision@{k}'] = np.nan

        for _, row in tqdm(self.eval_test.iterrows(), desc="Computing Precision@2", total=len(self.eval_test)):
            query = row['question']
            retrieved_docs = vector_db.search_similar_w_scores(query, k=k)
            retrieved_doc_ids = [doc[0].metadata['origin_doc_id'] for doc in retrieved_docs]
            ground_truth_id = row['top_score_id']

            relevant_docs = sum([1 for doc_id in retrieved_doc_ids[:k] if doc_id == ground_truth_id])
            precision = relevant_docs / k
            self.eval_test.at[_, f'precision@{k}'] = precision

    def compute_recall_at_k(self, vector_db, ks=[2]):
        k = ks[0]
        self.eval_test[f'recall@{k}'] = np.nan

        for _, row in tqdm(self.eval_test.iterrows(), desc="Computing Recall@2", total=len(self.eval_test)):
            query = row['question']
            retrieved_docs = vector_db.search_similar_w_scores(query, k=k)
            retrieved_doc_ids = [doc[0].metadata['origin_doc_id'] for doc in retrieved_docs]
            ground_truth_id = row['top_score_id']

            is_relevant_retrieved = int(ground_truth_id in retrieved_doc_ids[:k])
            recall = is_relevant_retrieved
            self.eval_test.at[_, f'recall@{k}'] = recall

    def plot_eval_result(self, df):

        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

        columns = [
            'faithfulness',
            'answer_relevancy',
            'context_recall',
            'context_precision',
            'context_entity_recall',
            'answer_similarity',
            'answer_correctness',
        ]

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[columns], palette="Set2", width=0.6, linewidth=1.5)
        plt.title(f"{self.name}: Ragas Metrics boxplot", fontsize=16)
        plt.ylabel("Score", fontsize=14)
        plt.xticks(fontsize=12, rotation=20)
        plt.tight_layout()
        plt.show()

    def plot_eval_result_bar(self, results):
        """
            Plot a barplot of evaluation scores for RAGAS metrics, MRR, precision@2, and recall@2.
            Args:
                results (pd.DataFrame): The results DataFrame from the evaluation.
            """
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

        ragas_metrics = [
            'faithfulness',
            'answer_relevancy',
            'context_precision',
            'context_recall',
            'context_entity_recall',
            'semantic_similarity',
            'answer_correctness',
        ]
        additional_metrics = ['MRR', 'precision@2', 'recall@2']
        all_metrics = ragas_metrics + additional_metrics

        metric_means = results[all_metrics].mean()

        plt.figure(figsize=(12, 6))
        sns.barplot(x=metric_means.index, y=metric_means.values, palette="Set2", hue=metric_means.index, legend=False)
        plt.title(f"{self.name}: Ragas + Non-LLM Metrics (Mean)", fontsize=16)
        plt.ylabel("Mean Score", fontsize=14)
        plt.xlabel("Metrics", fontsize=14)
        plt.xticks(fontsize=12, rotation=20)
        plt.tight_layout()
        plt.show()

    def plot_results_all(self, df, results):
        """
        Display both the boxplot for RAGAS metrics and the barplot for RAGAS + non-LLM metrics.
        Args:
            df (pd.DataFrame): DataFrame containing raw RAGAS metric values for the boxplot.
            results (pd.DataFrame): DataFrame containing RAGAS and non-LLM metric scores for the barplot.
        """
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

        # Boxplot: RAGAS Metrics
        boxplot_columns = [
            'faithfulness',
            'answer_relevancy',
            'context_precision',
            'context_entity_recall',
            'semantic_similarity',
            'answer_correctness',
        ]

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[boxplot_columns], palette="Set2", width=0.6, linewidth=1.5)
        plt.title(f"{self.name}: RAGAS Metrics Boxplot", fontsize=16)
        plt.ylabel("Score", fontsize=14)
        plt.xticks(fontsize=12, rotation=20)
        plt.tight_layout()
        plt.show()

        # Barplot: RAGAS + Non-LLM Metrics
        ragas_metrics = [
            'faithfulness',
            'answer_relevancy',
            'context_recall',
            'context_precision',
            'context_entity_recall',
            'semantic_similarity',
            'answer_correctness',
        ]
        additional_metrics = ['MRR', 'precision@2', 'recall@2']
        all_metrics = ragas_metrics + additional_metrics

        metric_means = results[all_metrics].mean()

        plt.figure(figsize=(12, 6))
        sns.barplot(x=metric_means.index, y=metric_means.values, palette="Set2", hue=metric_means.index, legend=False)
        plt.title(f"{self.name}: RAGAS + Non-LLM Metrics (Mean)", fontsize=16)
        plt.ylabel("Mean Score", fontsize=14)
        plt.xlabel("Metrics", fontsize=14)
        plt.xticks(fontsize=12, rotation=20)
        plt.tight_layout()
        plt.show()
