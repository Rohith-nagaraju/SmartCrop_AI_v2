from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'
EXPERIMENT_DIR = BASE_DIR / 'experiments'
DB_PATH = BASE_DIR / 'smartcrop.db'
IMAGE_SIZE = (224, 224)
UNKNOWN_THRESHOLD = 0.55
