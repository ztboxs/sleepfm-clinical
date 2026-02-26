import os
import json
import torch
from loguru import logger

from api.config import (
    BASE_MODEL_DIR,
    SLEEP_STAGING_MODEL_DIR,
    DISEASE_MODEL_DIR,
    CHANNEL_GROUPS_PATH,
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sleepfm"))
from models.models import SetTransformer, SleepEventLSTMClassifier, DiagnosisFinetuneFullLSTMCOXPHWithDemo


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _strip_module_prefix(state_dict: dict) -> dict:
    """Remove 'module.' prefix added by DataParallel."""
    new = {}
    for k, v in state_dict.items():
        new[k.removeprefix("module.")] = v
    return new


class ModelsManager:
    """Singleton-style manager that loads all three models once."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.channel_groups: dict = {}
        self.base_config: dict = {}
        self.sleep_staging_config: dict = {}
        self.disease_config: dict = {}
        self.base_model: SetTransformer | None = None
        self.sleep_staging_model: SleepEventLSTMClassifier | None = None
        self.disease_model: DiagnosisFinetuneFullLSTMCOXPHWithDemo | None = None

    def load_all(self):
        logger.info("Loading channel groups …")
        self.channel_groups = _load_json(CHANNEL_GROUPS_PATH)

        self._load_base_model()
        self._load_sleep_staging_model()
        self._load_disease_model()
        logger.info("All models loaded successfully.")

    def _load_base_model(self):
        logger.info("Loading base model (SetTransformer) …")
        cfg = _load_json(os.path.join(BASE_MODEL_DIR, "config.json"))
        self.base_config = cfg

        model = SetTransformer(
            in_channels=cfg["in_channels"],
            patch_size=cfg["patch_size"],
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            pooling_head=cfg["pooling_head"],
            dropout=0.0,
        )

        ckpt = torch.load(os.path.join(BASE_MODEL_DIR, "best.pt"), map_location=self.device)
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(_strip_module_prefix(state_dict))
        model.to(self.device).eval()
        self.base_model = model
        logger.info("Base model loaded.")

    def _load_sleep_staging_model(self):
        logger.info("Loading sleep staging model …")
        cfg = _load_json(os.path.join(SLEEP_STAGING_MODEL_DIR, "config.json"))
        self.sleep_staging_config = cfg

        params = cfg["model_params"]
        model = SleepEventLSTMClassifier(**params)

        ckpt = torch.load(os.path.join(SLEEP_STAGING_MODEL_DIR, "best.pth"), map_location=self.device)
        model.load_state_dict(_strip_module_prefix(ckpt))
        model.to(self.device).eval()
        self.sleep_staging_model = model
        logger.info("Sleep staging model loaded.")

    def _load_disease_model(self):
        logger.info("Loading disease prediction model …")
        cfg = _load_json(os.path.join(DISEASE_MODEL_DIR, "config.json"))
        self.disease_config = cfg

        params = dict(cfg["model_params"])
        params["dropout"] = 0.0
        model = DiagnosisFinetuneFullLSTMCOXPHWithDemo(**params)

        ckpt = torch.load(os.path.join(DISEASE_MODEL_DIR, "best.pth"), map_location=self.device)
        model.load_state_dict(_strip_module_prefix(ckpt))
        model.to(self.device).eval()
        self.disease_model = model
        logger.info("Disease prediction model loaded.")


manager = ModelsManager()
