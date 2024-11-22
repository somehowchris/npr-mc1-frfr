import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm


def plot_combined_boxplot(df):
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

    columns = [
        'faithfulness',
        'answer_relevancy',
        'context_precision',
        'context_entity_recall',
        'semantic_similarity',
        'answer_correctness',
    ]

    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=pd.melt(df, id_vars=['evaluation_name'], value_vars=columns),
        x='variable',
        y='value',
        hue='evaluation_name',
        palette="Set1",
        width=0.6,
        linewidth=1.5
    )
    plt.title("Ragas Metrics Boxplot by Evaluation", fontsize=16)
    plt.ylabel("Score", fontsize=14)
    plt.xlabel("Metrics", fontsize=14)
    plt.xticks(fontsize=12, rotation=20)
    plt.legend(title="Evaluation", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_combined_barplot(df):
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

    ragas_metrics = [
        'faithfulness',
        'answer_relevancy',
        'context_precision',
        'context_entity_recall',
        'semantic_similarity',
        'answer_correctness',
    ]
    additional_metrics = ['MRR', 'precision@2', 'recall@2']
    all_metrics = ragas_metrics + additional_metrics

    metric_means = df.groupby('evaluation_name')[all_metrics].mean()

    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=pd.melt(metric_means.reset_index(), id_vars=['evaluation_name'], var_name='metric',
                     value_name='mean_score'),
        x='metric',
        y='mean_score',
        hue='evaluation_name',
        palette="Set1"
    )
    plt.title("Mean Scores for Ragas and Non-LLM Metrics", fontsize=16)
    plt.ylabel("Mean Score", fontsize=14)
    plt.xlabel("Metrics", fontsize=14)
    plt.xticks(fontsize=12, rotation=20)
    plt.legend(title="Evaluation", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_compare_result(df):
    """
    Plot comparison results by generating both a bar plot and a box plot.
    Ensures consistent order of evaluation names in both plots and fixes warnings.

    Parameters:
        df (pd.DataFrame): DataFrame containing evaluation data.
    """
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

    # Define metrics
    ragas_metrics = [
        'faithfulness',
        'answer_relevancy',
        'context_precision',
        'context_entity_recall',
        'semantic_similarity',
        'answer_correctness',
    ]
    additional_metrics = ['MRR', 'precision@2', 'recall@2']
    all_metrics = ragas_metrics + additional_metrics

    # Ensure consistent ordering of 'evaluation_name'
    evaluation_order = df['evaluation_name'].unique()  # Get the unique order in the data
    df['evaluation_name'] = pd.Categorical(df['evaluation_name'], categories=evaluation_order, ordered=True)

    # Barplot
    metric_means = (
        df.groupby('evaluation_name', observed=False)[all_metrics]
        .mean()
        .reindex(evaluation_order)  # Align order
    )
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=pd.melt(metric_means.reset_index(), id_vars=['evaluation_name'], var_name='metric',
                     value_name='mean_score'),
        x='metric',
        y='mean_score',
        hue='evaluation_name',
        palette="Set1"
    )
    plt.title("Mean Scores for Ragas and Non-LLM Metrics", fontsize=16)
    plt.ylabel("Mean Score", fontsize=14)
    plt.xlabel("Metrics", fontsize=14)
    plt.xticks(fontsize=12, rotation=20)
    plt.legend(title="Evaluation", fontsize=12)
    plt.tight_layout()
    plt.show()

    # Boxplot
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=pd.melt(df, id_vars=['evaluation_name'], value_vars=ragas_metrics),
        x='variable',
        y='value',
        hue='evaluation_name',
        legend=False,
        palette="Set1",
        width=0.6,
        linewidth=1.5
    )
    plt.title("Ragas Metrics Boxplot by Evaluation", fontsize=16)
    plt.ylabel("Score", fontsize=14)
    plt.xlabel("Metrics", fontsize=14)
    plt.xticks(fontsize=12, rotation=20)
    plt.tight_layout()
    plt.show()


def create_documents(df: pd.DataFrame, text_splitter, verbose=True):
    metadata_cols = ['url', 'domain', 'title', 'date', 'id']
    if not all(col in df.columns for col in metadata_cols + ['content']):
        raise ValueError(
            f"DataFrame must contain all metadata columns and a 'content' column: {metadata_cols + ['content']}")

    metadata = df[metadata_cols].rename(columns={'id': 'origin_doc_id'}).to_dict('records')
    for i, m in enumerate(metadata):
        metadata[i] = {k: 'None' if v is None else v for k, v in m.items()}

    docs = text_splitter.create_documents(df['content'], metadata)

    if verbose:
        print(
            f"{text_splitter.__class__.__name__}: "
            f"Documents created: {len(docs)}, "
            f"Rows: {len(df)}, "
            f"Percentage of doc created: {len(docs) / len(df) * 100:.2f}%")

    return docs


def add_to_combined(eval_name, new_df, combined_df=None):
    new_df['evaluation_name'] = eval_name
    if combined_df is not None:
        return pd.concat([combined_df, new_df], ignore_index=True)
    else:
        return new_df
