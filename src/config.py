import logging
import os
from pathlib import Path


WORKING_DIR = Path(os.getenv('WORKING_DIR') or '.')
CONFIG_FILENAME = 'imput-config.yml'
CONFIG_PATH = Path(os.getenv('IMPUTATION_CONFIG_PATH') or WORKING_DIR / 'config' / CONFIG_FILENAME)

try:
    DICT_CONFIG = load_yaml(CONFIG_PATH)
    logging.info(f"Loaded config: {DICT_CONFIG}")
except FileNotFoundError:
    logging.warning(f"No config file found at {CONFIG_PATH}, assuming test mode")
    DICT_CONFIG = {}


class Config:
    """
    Configuration holder
    """
    # Common
    DEFAULT_DATEFORMAT = DICT_CONFIG.get('default_dateformat', '%Y-%m-%d')