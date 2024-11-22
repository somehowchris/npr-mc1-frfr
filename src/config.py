from pathlib import Path

# Resolve the project root directory dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the cache directory relative to the project root
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# Ensure the cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Define the persistent directory relative to the project root
PERSISTENT_DIR = PROJECT_ROOT / "data" / "chroma"

# Ensure the persistent directory exists
PERSISTENT_DIR.mkdir(parents=True, exist_ok=True)

# Optionally, expose a string version for compatibility with relative paths
PATH_PERSISTENT = str(PERSISTENT_DIR)

# Define the clean file path relative to the project root
CLEAN_FILE_PATH = PROJECT_ROOT / "data" / "preprocessed" / "clean_cleantech.parquet"

# Ensure the directory for the clean file exists
CLEAN_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Define the evaluation file path relative to the project root
EVAL_FILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "eval_dataset"
    / "cleantech_rag_evaluation_data_2024-09-20.csv"
)

# Ensure the directory for the evaluation file exists
EVAL_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL_HOST = "http://100.67.185.22"
