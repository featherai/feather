import os
import logging
from logging.handlers import RotatingFileHandler
try:
    from dotenv import load_dotenv
except Exception:
    # Fallback if python-dotenv is not installed
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

LOG_FILE = os.getenv("FEATHER_LOG_FILE", "feather.log")

def setup_logging(level=logging.INFO):
    logger = logging.getLogger("feather")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger

logger = setup_logging()


def safe_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def install_kaggle_credentials_from_env(dotenv_paths: list[str] | None = None) -> bool:
    """If KAGGLE_USERNAME and KAGGLE_KEY exist in environment or in .env/.env.example, write %USERPROFILE%\.kaggle\kaggle.json.

    Returns True if file was written or already exists, False otherwise.
    """
    try:
        import os
        import json

        home = os.path.expanduser('~')
        kaggle_dir = os.path.join(home, '.kaggle')
        kaggle_path = os.path.join(kaggle_dir, 'kaggle.json')

        # if already present, return True
        if os.path.exists(kaggle_path):
            return True

        username = os.environ.get('KAGGLE_USERNAME')
        key = os.environ.get('KAGGLE_KEY')

        # attempt to read from .env or .env.example if not in env
        if not username or not key:
            candidates = dotenv_paths or [os.path.join(os.path.dirname(__file__), '.env'), os.path.join(os.path.dirname(__file__), '.env.example')]
            for p in candidates:
                try:
                    if not os.path.exists(p):
                        continue
                    with open(p, 'r', encoding='utf-8') as fh:
                        for line in fh:
                            line = line.strip()
                            if line.startswith('KAGGLE_USERNAME') and ('=' in line):
                                username = username or line.split('=', 1)[1].strip()
                            if line.startswith('KAGGLE_KEY') and ('=' in line):
                                key = key or line.split('=', 1)[1].strip()
                except Exception:
                    continue

        if not username or not key:
            return False

        os.makedirs(kaggle_dir, exist_ok=True)
        # Do not log secrets
        payload = {"username": username, "key": key}
        with open(kaggle_path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh)
        try:
            # restrict permissions where possible
            os.chmod(kaggle_path, 0o600)
        except Exception:
            pass
        return True
    except Exception:
        logger.exception('Failed to write kaggle credentials from env')
        return False
