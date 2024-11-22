# Mini Challenge Retrieval Augmented Generation (RAG)

## Abstract
The mini-challenge focuses on the development and evaluation of a Retrieval Augmented Generation (RAG) system. This system enhances the capabilities of a language model by retrieving information from external sources to provide accurate answers. Participants are tasked with implementing and testing various components such as ingestion, retrieval, and generation modules, and will evaluate their system's performance using both human measures and automatic methods. The goal is to deepen understanding of NLP tools and improve practical skills in developing and evaluating advanced language models.

<!-- TOC -->
* [Mini Challenge Retrieval Augmented Generation (RAG)](#mini-challenge-retrieval-augmented-generation-rag)
  * [Abstract](#abstract)
  * [Project Structure](#project-structure)
    * [archiv](#archiv)
    * [data](#data)
    * [cache](#cache)
    * [notebooks](#notebooks)
    * [src](#src)
    * [assets](#assets)
  * [Getting Started](#getting-started)
    * [Installation](#installation)
<!-- TOC -->

## Project Structure

```
npr-mc1-frfr
├── archiv/
├── data
│   ├── assets
│   ├── cache
│   ├── chroma
│   ├── eval_dataset
│   ├── preprocessed
│   └── raw
├── notebooks
└── src
```

### archiv
This directory contains jupyter notebooks that were used once but got overwritten by other notebooks. They are kept for reference.

### data
This directory contains all data used in the project. It is divided into four subdirectories:

### cache
With this directory we used the `pickle` library to cache the results, evaluation, preprocessing which we could reuse later. With this approach we could save time and resources not rerunning the same code over and over again. The .pkl files aren't uploaded to the repository cause they can have conflicts based on python version used.  
The cacheing is handled directly from the `src.evaluation.py` file. The cache name is based on name of the evaluation the embedding model used llm model used and openai model used.  
For example: `cache/Baseline_qwen2.5:3b-instruct_BAAI_bge_m3_gpt-4o-mini_eval_result.pkl`

### notebooks
This directory contains all jupyter notebooks used in the project. All the experiments done the evaluation the exploration of the data preprocessing are in separate notebooks.

### src
Here we got all our scripts used in the project.  

`config.py` contains mainly the paths for the cache, raw data, preprocessed data and evaluation data.  

`custom_embeddings.py` created as a subclass from the package `HuggingfaceHubEndpointEmbeddings`. This was necessary because since we used local embeddings model with docker we couldn't get the model name form the model parameters. We added `model_name: Optional[str] = None` give the model the embeddings model name when initializing the class.  

`evaluation.py` contains the evaluation of the model. It also contains caching logic and plotting results. The `evaluate` method evaluates with the `ragas` package and the non-llm metrics: `precision@k`, `Mean Reciprocal Rank`, `Recall@k`.  

`preprocessing.py` contains the preprocessing of the data. It's a basic cleaner for the raw data. It detects language and removes non-english sentences. It removes every `<tags>` and with regex `re.sub(r'[^a-zA-Z0-9.,?!]', ' ', text)` it removes every special character. It also generates unique ids for each row.  

`utils.py` contains various helper functions and plotting functions. Also creates the complete combined results for the evaluation.  

`vectorstorage.py` is a subclass from the `Chromadb` package. This class manages a persistent vector storage for document embeddings using ChromaDB. It supports embedding, storing, retrieving, and managing document collections for similarity-based search.  

### assets
Here we stored Images if we had or used some.

## Getting Started
```
pip install -r requirements.txt
```
We used openai and local llm (Ollama) for generating answers. To use them you need to create a `.env` file in the root directory with the following content:
```
OPENAI_API_KEY=your_openai_api_key
```
For the LangSmith tracing and observing the System you need to insert this:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=your_api_key
LANGCHAIN_PROJECT=your_project_name
```
in your `.env` enviroment file.

And for local llms we used [Ollama](https://ollama.com). For this project we used the [Qwen2.5](https://ollama.com/library/qwen2.5) model. After installing ollama you can pull the model with the following command:
```
ollama run qwen2.5:3b-instruct
```
And calling the model with the following command:
```
from langchain_ollama import OllamaLLM
llm_model = OllamaLLM(model='qwen2.5:3b-instruct')
```
Since we ran the embeddings models local on a system with docker we had to use the following command to run the embeddings model:
```
docker-compose up
```
This sets up the docker container with the embeddings model.  
BAAI/bge_m3 with the port `8080`  
Nomic-embed-text-v1-5 with the port `8082`  
Qwen2-7b-instruct with the port `8083`

<!-- end of README.md -->


