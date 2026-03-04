import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLEEPFM_ROOT = os.path.join(PROJECT_ROOT, "sleepfm")

CHECKPOINT_DIR = os.path.join(SLEEPFM_ROOT, "checkpoints")
BASE_MODEL_DIR = os.path.join(CHECKPOINT_DIR, "model_base")
SLEEP_STAGING_MODEL_DIR = os.path.join(CHECKPOINT_DIR, "model_sleep_staging")
DISEASE_MODEL_DIR = os.path.join(CHECKPOINT_DIR, "model_diagnosis")

CHANNEL_GROUPS_PATH = os.path.join(SLEEPFM_ROOT, "configs", "channel_groups.json")
LABEL_MAPPING_PATH = os.path.join(SLEEPFM_ROOT, "configs", "label_mapping.csv")

RESAMPLE_RATE = 128
MAX_UPLOAD_SIZE_MB = 0  # 0 means no limit

EXPORT_DIR = os.path.join(PROJECT_ROOT, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)
EXPORT_MAX_AGE_HOURS = 24

API_HOST = "0.0.0.0"
API_PORT = 6006
