import os
import json
from utils import logger, safe_cuda_available


def run_sample_pipeline(sample_text: str):
    logger.info("Running sample pipeline")
    cuda = safe_cuda_available()
    logger.info(f"CUDA available: {cuda}")
    # placeholder
    out = {"risk": "green", "summary": "No issues detected.", "sources": []}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run_sample_pipeline("Test news article about markets")
