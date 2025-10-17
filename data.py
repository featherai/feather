import os
import pandas as pd
import logging
from utils import logger

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def load_csv_safe(path: str, nrows: int | None = None) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, nrows=nrows)
        return df
    except Exception as e:
        logger.exception("Failed to load CSV %s", path)
        raise


def save_preprocessed(df: pd.DataFrame, name: str):
    out_parquet = os.path.join(DATA_DIR, f"{name}.parquet")
    out_csv = os.path.join(DATA_DIR, f"{name}.csv")
    try:
        # try parquet first
        df.to_parquet(out_parquet)
        logger.info("Saved preprocessed data to %s", out_parquet)
        return out_parquet
    except Exception:
        logger.warning("Parquet write failed, falling back to CSV for %s", name)
        try:
            df.to_csv(out_csv, index=False)
            logger.info("Saved preprocessed data to %s", out_csv)
            return out_csv
        except Exception:
            logger.exception("Failed to save preprocessed data for %s", name)
            raise


def subset_df(df: pd.DataFrame, max_rows: int = 10000) -> pd.DataFrame:
    try:
        if len(df) > max_rows:
            return df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        return df.reset_index(drop=True)
    except Exception:
        logger.exception("Failed to subset dataframe")
        raise


def download_kaggle_dataset(dataset: str, file_name: str, dest: str | None = None) -> str:
    """Attempt to download a dataset using kaggle CLI. Returns path to file.

    dataset: e.g., 'ahmedtallal/ISOT-Fake-News-Dataset'
    file_name: the csv file inside the dataset
    """
    try:
        dest = dest or DATA_DIR
        out = os.path.join(dest, file_name)
        # Ensure kaggle credentials exist in ~/.kaggle/kaggle.json if possible
        try:
            from utils import install_kaggle_credentials_from_env
            install_kaggle_credentials_from_env()
        except Exception:
            pass

        # Try to import kaggle python client; if not available, call kaggle CLI
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            # Some kaggle client versions do not accept an `unzip` kwarg for
            # dataset_download_file. Call without unzip and handle any zip file
            # produced by the server manually.
            api.dataset_download_file(dataset, file_name, path=dest)
            # If the API saved a zipped file, try to unzip it
            possible_zip = os.path.join(dest, f"{file_name}.zip")
            if os.path.exists(possible_zip):
                try:
                    import zipfile
                    import tempfile

                    # Use a temporary directory for extraction to avoid partial extracts
                    with tempfile.TemporaryDirectory() as tmpdir:
                        with zipfile.ZipFile(possible_zip, 'r') as zf:
                            # Check for path traversal attacks
                            for fname in zf.namelist():
                                if fname.startswith('/') or '..' in fname:
                                    raise ValueError(f"Invalid path in zip: {fname}")
                            # Extract to temp dir first
                            zf.extractall(tmpdir)
                            # Move successful extracts to final destination
                            import shutil
                            for fname in os.listdir(tmpdir):
                                src = os.path.join(tmpdir, fname)
                                dst = os.path.join(dest, fname)
                                shutil.move(src, dst)
                    # Only remove zip after successful extraction
                    os.remove(possible_zip)
                except Exception as e:
                    logger.warning("Failed to unzip %s: %s", possible_zip, e)
            return out
        except Exception as e:
            logger.warning("kaggle python client not available or failed: %s", e)
            # fallback to system kaggle CLI if available
            import shutil, subprocess

            kaggle_cli = shutil.which('kaggle')
            if kaggle_cli is None:
                logger.error("kaggle CLI not found. Install the kaggle package (pip install kaggle) and/or the kaggle CLI and add credentials.")
                raise
            # Use the CLI as a fallback. --unzip is helpful but some servers
            # may return HTTP 403 if the dataset requires acceptance or the
            # account lacks permission; log clear guidance in that case.
            cmd = [kaggle_cli, "datasets", "download", "-d", dataset, "-f", file_name, "-p", dest, "--unzip"]
            logger.info("Running: %s", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as cpe:
                logger.error("kaggle CLI download failed: %s", cpe)
                logger.error("If you see a 403 Forbidden, visit the dataset page and accept the terms or verify your API token at %s", os.path.join(os.path.expanduser('~'), '.kaggle', 'kaggle.json'))
                raise
            return out
    except Exception:
        logger.exception("Failed to download dataset %s/%s", dataset, file_name)
        raise


def generate_synthetic_news(n: int = 2000) -> pd.DataFrame:
    """Create a tiny synthetic news dataset for local testing."""
    try:
        import random

        rows = []
        for i in range(n):
            label = random.choice(["verified", "misleading"]) if i % 3 == 0 else "verified"
            txt = f"Sample news article {i} about market movements."
            if label == "misleading":
                txt += " This contains fake claims and rumors."
            rows.append({"title": f"Title {i}", "text": txt, "label": label})
        df = pd.DataFrame(rows)
        return df
    except Exception:
        logger.exception("Failed to generate synthetic data")
        raise


def prepare_isot_dataset(dest: str | None = None, max_rows: int = 10000) -> str:
    """Prepare an ISOT dataset file for training without relying on Kaggle.

    - If True.csv and Fake.csv exist under DATA_DIR, use them. Otherwise fall back
      to a synthetic dataset and save as isot_subset_10k.parquet/csv.
    - Returns the path to the saved preprocessed file.
    """
    try:
        dest = dest or DATA_DIR
        true_path = os.path.join(dest, 'True.csv')
        fake_path = os.path.join(dest, 'Fake.csv')

        dfs = []
        if os.path.exists(true_path) and os.path.exists(fake_path):
            logger.info('Found local True.csv and Fake.csv in %s; using local files', dest)
            try:
                tdf = load_csv_safe(true_path)
                fdf = load_csv_safe(fake_path)
                for df, label in [(tdf, 0), (fdf, 1)]:
                    if 'text' not in df.columns:
                        for c in ['text', 'Content', 'content', 'article']:
                            if c in df.columns:
                                df = df.rename(columns={c: 'text'})
                                break
                    if 'text' not in df.columns:
                        logger.warning('No text column found in one of the provided files; skipping')
                        continue
                    df = df[['text']].copy()
                    df['label'] = label
                    dfs.append(df)
            except Exception:
                logger.exception('Failed to read local True/Fake CSVs; falling back to synthetic')

        if not dfs:
            logger.info('Local ISOT files not available or unreadable; generating synthetic dataset')
            synth = generate_synthetic_news(n=max_rows)
            out_path = save_preprocessed(synth, 'isot_subset_10k')
            return out_path

        all_df = pd.concat(dfs, ignore_index=True)
        logger.info('Combined local dataset rows: %d', len(all_df))
        if len(all_df) > max_rows:
            all_df = subset_df(all_df, max_rows)
        out = save_preprocessed(all_df, 'isot_subset_10k')
        logger.info('ISOT preprocessed saved to %s', out)
        return out
    except Exception:
        logger.exception('prepare_isot_dataset failed; generating synthetic fallback')
        synth = generate_synthetic_news(n=max_rows)
        out_path = save_preprocessed(synth, 'isot_subset_10k')
        return out_path
