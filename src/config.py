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