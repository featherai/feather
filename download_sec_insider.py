import sys
import pathlib
import requests
import zipfile
import io
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import logger

def download_sec_insider_data(year: int = 2024, quarter: int = 3) -> pd.DataFrame:
    """
    Download SEC Form 4 insider transactions data for a specific quarter.
    Returns DataFrame with insider filings.
    """
    url = f"https://www.sec.gov/files/dera/data/form-4-insider-transactions-data-sets/{year}q{quarter}_form4.zip"
    logger.info(f"Downloading SEC insider data from {url}")
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        
        # Extract ZIP
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            # Find the XML or CSV file
            files = z.namelist()
            xml_file = [f for f in files if f.endswith('.xml')][0]
            with z.open(xml_file) as f:
                # Parse XML to DataFrame
                import xml.etree.ElementTree as ET
                tree = ET.parse(f)
                root = tree.getroot()
                
                data = []
                for filing in root.findall('.//Filing'):
                    filing_data = {}
                    for child in filing:
                        filing_data[child.tag] = child.text
                    data.append(filing_data)
                
                df = pd.DataFrame(data)
                logger.info(f"Loaded {len(df)} insider filings")
                return df
    except Exception as e:
        logger.exception(f"Failed to download SEC data: {e}")
        return pd.DataFrame()

def prepare_insider_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process SEC data to get daily insider activity per symbol.
    """
    if df.empty:
        return df
    
    # Convert dates
    df['transactionDate'] = pd.to_datetime(df['transactionDate'], errors='coerce')
    df['filingDate'] = pd.to_datetime(df['filingDate'], errors='coerce')
    
    # Filter to buys/sells
    df = df[df['transactionType'].isin(['Buy', 'Sell'])]
    
    # Group by symbol and date, sum net value
    df['value'] = df['transactionShares'].astype(float) * df['transactionPrice'].astype(float)
    df['net_value'] = df.apply(lambda r: r['value'] if r['transactionType'] == 'Buy' else -r['value'], axis=1)
    
    grouped = df.groupby(['issuerTickerSymbol', 'transactionDate'])['net_value'].sum().reset_index()
    grouped.rename(columns={'issuerTickerSymbol': 'symbol', 'transactionDate': 'date'}, inplace=True)
    
    return grouped

def main():
    df = download_sec_insider_data(2024, 3)
    if not df.empty:
        insider_df = prepare_insider_dataset(df)
        out_path = ROOT / "data" / "insider_activity_2024q3.csv"
        out_path.parent.mkdir(exist_ok=True)
        insider_df.to_csv(out_path, index=False)
        logger.info(f"Saved insider dataset to {out_path}")
    else:
        logger.error("No data downloaded")

if __name__ == "__main__":
    main()
