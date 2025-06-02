import os
import sys
import logging

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Base app configuration
APP_DIR = Path(__file__).parent
TEMP_DIR = APP_DIR / "temp"
RESULT_DIR = APP_DIR / "results"
LOG_FILE = APP_DIR / "app.log"

# Crop configuration
ORDERED_INDICES = [0, 1, 2, 5, 8, 11, 14, 17, 20, 19, 18, 15, 12, 9, 6, 3, 0]
TARGET_WIDTH = 256
TARGET_HEIGHT = 384

if LOG_FILE.exists():
    os.remove(LOG_FILE)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)