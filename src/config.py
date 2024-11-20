from pathlib import Path

# Resolve the project root directory dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the cache directory relative to the project root
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# Ensure the cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)