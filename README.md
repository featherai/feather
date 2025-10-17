# Feather AI: Advanced Asset Risk Intelligence

Feather AI is a comprehensive Python framework for real-time financial asset analysis. It combines dual-head machine learning models, news sentiment analysis with topic modeling and event detection, rule-based risk assessment, and on-chain crypto metrics to provide holistic risk and direction signals.

## Features

- **Dual-Head ML Model**: Predicts drawdown risk and upward price direction using 14+ technical features.
- **News Analysis**: Sentiment via FinBERT/Transformers, topic modeling (LDA), and event detection (earnings, mergers, etc.).
- **Risk Rules**: Green/Yellow/Red risk levels based on volatility and drawdowns.
- **On-chain Crypto**: Web3 integration for Ethereum gas prices and large transfers (demo functions).
- **Outputs**: Jaw-dropping formatted CLI and interactive Streamlit dashboard.
- **Backtesting**: Historical signal simulation with performance metrics.

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/your-repo/feather-ai.git
   cd feather-ai
   ```

2. Create virtual env:
   ```bash
   python -m venv feather_env
   source feather_env/bin/activate  # On Windows: feather_env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables: Copy `.env.example` to `.env` and fill in API keys:
   - `POLYGON_API_KEY`: For stock data (free tier available).
   - `COINGECKO_API_KEY`: For crypto data.
   - `NEWSAPI_KEY`: For news (free tier at newsapi.org).
   - `ETHERSCAN_API_KEY`: For Ethereum on-chain data.

## Usage

### Web App (Easiest)
Run the interactive web interface:
```bash
python flask_app.py
```
Open http://localhost:5000 – enter crypto address/stock ticker, get instant analysis.

### Streamlit Dashboard
Train on 30+ assets (stocks + crypto):
```bash
python scripts/train_asset_dual_model.py --tickers AAPL,MSFT,BTC-USD,ETH-USD --period 5y --epochs 100
```

### Analyze Assets
- **CLI (BTC)**: `python scripts/analyze_btc_now.py`
- **Dual Inference**: `python scripts/smoke_infer_dual.py --ticker AAPL`
- **Backtest**: `python scripts/backtest_dual.py --symbol BTC/USDT --period 1y`

### Dashboard
Launch interactive app:
```bash
streamlit run scripts/app_streamlit.py --server.port 8501
```
- Enter ticker, view charts, probabilities, drivers, news sentiment, topics, events.

### API Keys
- **Polygon.io**: Sign up at polygon.io for API key (5M agg req/month free).
- **CoinGecko**: Get free API key at coingecko.com.
- **NewsAPI**: Optional, free tier at newsapi.org.
- **Etherscan**: For Web3, get API key at etherscan.io.

## Architecture

```
feather/
├── models/          # Trained PyTorch models
├── scripts/         # CLI scripts and dashboard
├── fetchers.py      # Data fetching (stocks, crypto, news, on-chain)
├── asset_model.py   # Dual-head NN model
├── sentiment.py     # News analysis with topics/events
├── risks.py         # Rule-based risk scoring
└── utils.py         # Helpers
```

## Key Components

- **ML Model**: Trained on 5y data, AUC down=0.74, up=0.53.
- **News**: Sentiment + LDA topics + keyword events.
- **Web3 Integration**: Gas fees, large transactions (demo in fetchers.py).
- **Backtesting**: Simulates trades, calculates returns/Shapre/MDD.

## Performance

- **Training**: 100 epochs, 30 tickers, ~1 hour.
- **Inference**: <1s per asset.
- **Backtest (BTC 1y)**: 33.77% return, Sharpe 0.78.

## Contributing

- Add more features: Options, forex, or advanced ML.
- Improve models: Fine-tune on labeled data.
- Expand data: More APIs, real-time streams.

## License

MIT License.

Powered by Feather AI – Smarter Asset Decisions! 🚀
