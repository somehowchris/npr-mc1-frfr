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

nest_asyncio.apply()

class RAGEvaluation:
    def __init__(self, name, rag_chain, llm_model, embeddings):
        self.name = name
        self.rag_chain = rag_chain
        self.llm_model = llm_model
        self.embeddings = embeddings
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
        """
        Generate dynamic filenames for cache files.
        Args:
            base_name (str): Base name for the cache file.
            step (str): Step in the pipeline (e.g., 'preprocess', 'dataset', 'evaluation').
        Returns:
            str: The generated cache filename.
        """
        embedding_model_name = getattr(self.embeddings, "model_name", "unknown_embedding_model")

        if step == "preprocess":
            return f"data/cache/{base_name}.pkl"  # Static for preprocessing
        elif step == "dataset":
            return f"data/cache/{embedding_model_name}_dataset.pkl"
        elif step == "evaluation":
            global_llm_name = getattr(self.llm_model, "model_name", "unknown_global_llm")
            return f"data/cache/{embedding_model_name}_{global_llm_name}_eval_result.pkl"
        else:
            raise ValueError(f"Unknown step: {step}")

    def preprocess(self, clean_file, eval_file):
        cache_file = self.get_dynamic_filename("preprocessed", "preprocess")
        cache_path = Path(cache_file)

        # Check for cached data
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

        # Check for cached dataset
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
            retrieved_contexts = [doc.page_content for doc in result_chain["context"]]
            data["user_input"].append(query)
            data["retrieved_contexts"].append(retrieved_contexts)
            data["response"].append(self.rag_chain.invoke(query)['answer'])

        self.dataset = Dataset.from_dict(data)
        self.save_to_cache(self.dataset, cache_file)

    async def async_adapt_prompts(self, llm_wrapper):
        await faithfulness.adapt_prompts(language="english", llm=llm_wrapper)

    def evaluate(self, clean_file_path, eval_file_path):
        """
        Evaluates the RAG system after preprocessing and preparing the dataset.
        Args:
            clean_file_path (str): Path to the clean dataset file.
            eval_file_path (str): Path to the evaluation file.
        Returns:
            pd.DataFrame: Evaluation results as a Pandas DataFrame.
        """
        cache_file = self.get_dynamic_filename("evaluation_result", "evaluation")
        cache_path = Path(cache_file)

        # Check for cached evaluation results
        if cache_path.exists():
            self.evaluation_result = self.load_from_cache(cache_file)
            if isinstance(self.evaluation_result, pd.DataFrame):
                return self.evaluation_result
            elif hasattr(self.evaluation_result, 'to_pandas'):
                return self.evaluation_result.to_pandas()
            else:
                raise TypeError("Cached evaluation result is not in a compatible format.")

        # Preprocess and prepare dataset
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

        # Perform evaluation
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

        # Save the serialized dictionary to the cache
        self.save_to_cache(result, cache_file)

        print("Evaluation complete.")
        return result

    def plot_eval_result(self, df):
        columns = [
            #'faithfulness',
            'answer_relevancy',
            'context_precision',
            'context_entity_recall',
            'semantic_similarity',
            'answer_correctness',
        ]
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[columns], palette="Set2", width=0.6, linewidth=1.5)
        plt.title(f"Boxplot of Eval Scores {self.name}", fontsize=16)
        plt.ylabel("Score", fontsize=14)
        plt.xticks(fontsize=12, rotation=20)
        plt.tight_layout()
        plt.show()

