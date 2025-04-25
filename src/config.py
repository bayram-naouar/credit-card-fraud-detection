from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW = ROOT_DIR / '..' /'data' / 'raw' / 'creditcard.csv'
DATA_PROCESSED_DIR = ROOT_DIR / '..' / 'data' / 'processed'
MODELS_DIR = ROOT_DIR / '..' / 'models'