import os

# Default paths (can be overridden by environment variables)
DATA_DIR = os.environ.get('DATA_DIR', 'data')
METADATA_CSV = os.environ.get('METADATA_CSV', os.path.join(DATA_DIR, 'metadata.csv'))
SKIN_METADATA_CSV = os.environ.get('SKIN_METADATA_CSV', os.path.join(DATA_DIR, 'skin_metadata.csv'))
COMBINED_DATASET_DIR = os.environ.get('COMBINED_DATASET_DIR', os.path.join(DATA_DIR, 'combined_shuffled_dataset_dict'))

# kdb.ai config
KDBAI_ENDPOINT = os.environ.get('KDBAI_ENDPOINT')
KDBAI_API_KEY = os.environ.get('KDBAI_API_KEY')
