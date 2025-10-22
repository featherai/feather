import logging
import os
import json
from utils import logger
from typing import Dict, Any

# Default thresholds for risk scoring; can be overridden by a JSON file pointed to by
# FEATHER_RISK_CONFIG env var or a local file named 'risk_config.json' in project root.
DEFAULT_THRESHOLDS = {
        "vol_high": 0.10,
        "vol_mid": 0.03,
        "atr_high": 0.08,
        "atr_mid": 0.04,
        "mdd_high": -0.30,
        "mdd_mid": -0.15,
        "rsi_overbought": 75.0,
        "rsi_oversold": 25.0,
        "mom_neg": -0.10,
        "vol_spike": 2.0,
        "holder_conc_high": 0.5,
        # Advanced metrics thresholds
        "beta_high": 1.5,          # high systematic risk
        "sharpe_low_yellow": 0.0,  # non-positive Sharpe
        "sharpe_low_red": -0.5,    # deeply negative Sharpe
        "var95_deep": -0.05,       # 5% VaR (<= threshold is worse)
        "cvar95_deep": -0.08,      # expected shortfall
        "red_cutoff": 5.0,
        "yellow_cutoff": 3.0,
}

_THRESH_CACHE: Dict[str, float] | None = None


def get_thresholds() -> Dict[str, float]:
    global _THRESH_CACHE
    if _THRESH_CACHE is not None:
        return _THRESH_CACHE
    # Try env var first
    cfg_path = os.environ.get("FEATHER_RISK_CONFIG")
    candidates = []
    if cfg_path and os.path.exists(cfg_path):
        candidates.append(cfg_path)
    # Project root risk_config.json
    here = os.path.dirname(__file__)
    candidates.append(os.path.join(here, "risk_config.json"))
    # Also check parent dir (project root if this file is in a package)
    candidates.append(os.path.join(os.path.dirname(here), "risk_config.json"))

    thresholds = DEFAULT_THRESHOLDS.copy()
    for p in candidates:
        try:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if k in thresholds:
                                try:
                                    thresholds[k] = float(v)
                                except Exception:
                                    continue
                break
        except Exception:
            # Ignore config errors and use defaults
            continue
    _THRESH_CACHE = thresholds
    return thresholds


def rule_based_risk_score(features: Dict[str, Any]) -> Dict[str, Any]:
    """Map computed features to a risk score, bucket, and reasons.

    Signals considered:
    - Volatility via returns std and ATR percent
    - Max drawdown severity
    - RSI extremes
    - Short-term momentum
    - Volume spike ratio
    - Holder concentration (if provided)
    - Misinfo score (if provided; de-emphasized by default)
    """
    try:
        thr = get_thresholds()
        score = 0
        reasons = []
        vol = float(features.get("volatility", 0) or 0)
        atr_pct = float(features.get("atr_pct", 0) or 0)
        if vol > thr["vol_high"] or atr_pct > thr["atr_high"]:
            score += 2
            reasons.append("High volatility")
        elif vol > thr["vol_mid"] or atr_pct > thr["atr_mid"]:
            score += 1
            reasons.append("Medium volatility")

        mdd = float(features.get("max_drawdown", 0) or 0)  # negative number
        if mdd < thr["mdd_high"]:
            score += 2
            reasons.append("Large drawdown")
        elif mdd < thr["mdd_mid"]:
            score += 1
            reasons.append("Moderate drawdown")

        rsi14 = float(features.get("rsi_14", 50) or 50)
        if rsi14 >= thr["rsi_overbought"]:
            score += 1
            reasons.append("Overbought (RSI)")
        elif rsi14 <= thr["rsi_oversold"]:
            score += 1
            reasons.append("Oversold (RSI)")

        mom10 = float(features.get("momentum_10", 0) or 0)
        if mom10 <= thr["mom_neg"]:
            score += 1
            reasons.append("Negative momentum")

        vol_ratio = float(features.get("volume_ratio_3_20", 0) or 0)
        if vol_ratio >= thr["vol_spike"]:
            score += 1
            reasons.append("Volume spike")

        # Advanced metrics: beta, sharpe, VaR/ES
        beta = features.get("beta_vs_spy", None)
        if beta is not None:
            try:
                if float(beta) >= thr["beta_high"]:
                    score += 1
                    reasons.append("High market beta")
            except Exception:
                pass

        sharpe = features.get("sharpe_30d", None)
        if sharpe is not None:
            try:
                s = float(sharpe)
                if s <= thr["sharpe_low_red"]:
                    score += 2
                    reasons.append("Very poor Sharpe")
                elif s <= thr["sharpe_low_yellow"]:
                    score += 1
                    reasons.append("Low Sharpe")
            except Exception:
                pass

        var95 = features.get("var_95", None)
        if var95 is not None:
            try:
                if float(var95) <= thr["var95_deep"]:
                    score += 1
                    reasons.append("High VaR tail risk")
            except Exception:
                pass

        cvar95 = features.get("cvar_95", None)
        if cvar95 is not None:
            try:
                if float(cvar95) <= thr["cvar95_deep"]:
                    score += 1
                    reasons.append("High CVaR tail risk")
            except Exception:
                pass

        holder_concentration = features.get("holder_concentration", 0)
        if holder_concentration > thr["holder_conc_high"]:
            score += 2
            reasons.append("High holder concentration")

        # De-emphasized misinfo in asset-only usage, keep for compatibility
        news_misinfo = float(features.get("misinfo_score", 0) or 0)
        if news_misinfo > 0.8:
            score += 1  # reduced weight
            reasons.append("Misinformation detected")
        elif news_misinfo > 0.5:
            score += 0.5
            reasons.append("Suspicious claims")

        # normalize
        if score >= thr["red_cutoff"]:
            bucket = "red"
        elif score >= thr["yellow_cutoff"]:
            bucket = "yellow"
        else:
            bucket = "green"

        summary = ", ".join(reasons)[:300]
        return {"bucket": bucket, "score": score, "reasons": reasons, "summary": summary}
    except Exception:
        logger.exception("Failed to compute rule-based risk")
        raise


def extract_basic_features_from_ohlcv(df) -> Dict[str, Any]:
    """Compute features from OHLCV or price series.

    Accepts either a pandas DataFrame with OHLCV columns or a list of
    [timestamp, open, high, low, close, volume]. If only a price series is
    available (e.g., 'price' column), computes a subset of features.
    """
    try:
        import numpy as np
        import pandas as pd

        if df is None:
            return {"volatility": 0.0, "recent_return": 0.0, "atr_pct": 0.0, "max_drawdown": 0.0, "rsi_14": 50.0, "momentum_10": 0.0, "volume_ratio_3_20": 0.0}
        if isinstance(df, list):
            # convert to close series
            closes = [r[4] for r in df if len(r) >= 5]
            ser = pd.Series(closes, dtype=float)
            # For list, we can compute ATR from open/high/low if present
            try:
                highs = pd.Series([r[2] for r in df], dtype=float)
                lows = pd.Series([r[3] for r in df], dtype=float)
                prev_close = ser.shift(1)
                tr = pd.concat([
                    (highs - lows).abs(),
                    (highs - prev_close).abs(),
                    (lows - prev_close).abs(),
                ], axis=1).max(axis=1)
                atr = tr.rolling(window=14, min_periods=1).mean()
                atr_pct = float((atr.iloc[-1] / ser.iloc[-1])) if len(ser) > 0 else 0.0
            except Exception:
                atr_pct = 0.0
        else:
            cols = {c.lower(): c for c in df.columns}
            close_col = cols.get("close") or cols.get("adj close") or cols.get("price")
            if close_col is None:
                # Fallback to last column heuristic
                close_col = df.columns[-1]
            ser = pd.Series(df[close_col], dtype=float)

            # Advanced metrics if OHLC available
            atr_pct = 0.0
            try:
                high_col = cols.get("high")
                low_col = cols.get("low")
                if high_col and low_col:
                    highs = pd.Series(df[high_col], dtype=float)
                    lows = pd.Series(df[low_col], dtype=float)
                    prev_close = ser.shift(1)
                    tr = pd.concat([
                        (highs - lows).abs(),
                        (highs - prev_close).abs(),
                        (lows - prev_close).abs(),
                    ], axis=1).max(axis=1)
                    atr = tr.rolling(window=14, min_periods=1).mean()
                    atr_pct = float((atr.iloc[-1] / ser.iloc[-1])) if len(ser) > 0 else 0.0
            except Exception:
                atr_pct = 0.0
        returns = ser.pct_change().dropna()
        vol = float(returns.std()) if not returns.empty else 0.0
        recent_return = float((ser.iloc[-1] / ser.iloc[0] - 1)) if len(ser) > 1 else 0.0

        # Max drawdown
        try:
            roll_max = ser.cummax()
            drawdown = ser / roll_max - 1.0
            max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
        except Exception:
            max_drawdown = 0.0

        # RSI (14)
        def _rsi(s: pd.Series, period: int = 14) -> float:
            delta = s.diff()
            up = delta.clip(lower=0.0)
            down = -delta.clip(upper=0.0)
            roll_up = up.rolling(window=period, min_periods=period).mean()
            roll_down = down.rolling(window=period, min_periods=period).mean()
            rs = roll_up / (roll_down.replace(0, 1e-12))
            rsi = 100.0 - (100.0 / (1.0 + rs))
            if rsi.dropna().empty:
                return 50.0
            return float(rsi.iloc[-1])

        try:
            rsi_14 = _rsi(ser, 14)
        except Exception:
            rsi_14 = 50.0

        # Momentum (10-day ROC)
        try:
            if len(ser) > 10 and ser.iloc[-10] != 0:
                momentum_10 = float(ser.iloc[-1] / ser.iloc[-10] - 1.0)
            else:
                momentum_10 = 0.0
        except Exception:
            momentum_10 = 0.0

        # Volume ratio (last 3 avg vs last 20 avg) if volume available
        try:
            vol_ratio = 0.0
            if isinstance(df, list):
                vols = pd.Series([r[5] for r in df if len(r) >= 6], dtype=float)
            else:
                vcol = cols.get("volume") if not isinstance(df, list) else None
                vols = pd.Series(df[vcol], dtype=float) if vcol else None
            if vols is not None and len(vols) >= 5:
                ma3 = vols.rolling(window=3, min_periods=1).mean()
                ma20 = vols.rolling(window=20, min_periods=1).mean()
                vol_ratio = float((ma3.iloc[-1] / (ma20.iloc[-1] + 1e-12)))
            volume_ratio_3_20 = vol_ratio
        except Exception:
            volume_ratio_3_20 = 0.0

        # Advanced metrics (optional: set FEATHER_ADVANCED_METRICS=1)
        beta_vs_spy = None
        sharpe_30d = None
        var_95 = None
        cvar_95 = None
        try:
            if os.environ.get("FEATHER_ADVANCED_METRICS", "0") == "1":
                # Sharpe over last 30 returns
                if len(returns) >= 5:
                    tail = returns.tail(30)
                    mu = float(tail.mean())
                    sd = float(tail.std())
                    sharpe_30d = float(mu / (sd + 1e-12)) if sd > 0 else 0.0
                    # VaR / CVaR (95%)
                    q = float(tail.quantile(0.05))
                    var_95 = q
                    cvar_95 = float(tail[tail <= q].mean()) if (tail <= q).any() else q
                # Beta vs SPY using overlapping returns
                try:
                    from fetchers import fetch_stock_history_yfinance  # local import to avoid cycles
                    ref = fetch_stock_history_yfinance("SPY", period="6mo")
                    if ref is not None and not ref.empty and len(returns) > 5:
                        ref_ret = ref["Close"].pct_change().dropna()
                        joint = pd.concat([returns, ref_ret], axis=1, join="inner").dropna()
                        joint.columns = ["asset", "spy"]
                        if len(joint) > 5 and float(joint["spy"].var()) > 0:
                            beta_vs_spy = float(joint.cov().iloc[0, 1] / joint["spy"].var())
                except Exception:
                    pass
        except Exception:
            pass

        return {
            "volatility": vol,
            "recent_return": recent_return,
            "atr_pct": atr_pct,
            "max_drawdown": max_drawdown,
            "rsi_14": rsi_14,
            "momentum_10": momentum_10,
            "volume_ratio_3_20": volume_ratio_3_20,
            "beta_vs_spy": beta_vs_spy,
            "sharpe_30d": sharpe_30d,
            "var_95": var_95,
            "cvar_95": cvar_95,
        }
    except Exception:
        logger.exception("Failed to extract features from ohlcv")
        return {"volatility": 0.0, "recent_return": 0.0, "atr_pct": 0.0, "max_drawdown": 0.0, "rsi_14": 50.0, "momentum_10": 0.0, "volume_ratio_3_20": 0.0}


def assess_asset_risk(ohlcv, misinfo_score: float = 0.0, holder_concentration: float = 0.0) -> Dict[str, Any]:
    try:
        feats = extract_basic_features_from_ohlcv(ohlcv)
        feats["misinfo_score"] = misinfo_score
        feats["holder_concentration"] = holder_concentration
        risk = rule_based_risk_score(feats)
        return {"features": feats, "risk": risk}
    except Exception:
        logger.exception("Failed to assess asset risk")
        return {"features": {}, "risk": {"bucket": "green", "score": 0}}


def analyze_asset(symbol: str, period: str = "1mo", holder_concentration: float = 0.0) -> Dict[str, Any]:
    """Convenience wrapper to fetch data and return an assessment.

    Stocks are fetched via yfinance. For crypto, prefer preprocessed OHLC data.
    """
    try:
        from fetchers import fetch_stock_history_yfinance, fetch_high_quality_crypto_data

        if "/" in symbol:
            # Map common tickers to CoinGecko IDs when given pairs like "BTC/USDT"
            base = symbol.split("/", 1)[0].upper().strip()
            cg_map = {
                "BTC": "bitcoin",
                "ETH": "ethereum",
                "BNB": "binancecoin",
                "XRP": "ripple",
                "ADA": "cardano",
                "SOL": "solana",
                "DOGE": "dogecoin",
                "DOT": "polkadot",
                "MATIC": "matic-network",
                "LTC": "litecoin",
            }
            coin_id = cg_map.get(base, base.lower())
            prices = fetch_high_quality_crypto_data(coin_id)
            ohlcv_like = prices  # price-only; advanced features will partially compute
        else:
            ohlcv_like = fetch_stock_history_yfinance(symbol, period=period)

        if ohlcv_like is None:
            return {"features": {}, "risk": {"bucket": "green", "score": 0, "reasons": ["No data fetched"]}}

        assessment = assess_asset_risk(ohlcv_like, misinfo_score=0.0, holder_concentration=holder_concentration)

        # Attach ML inference if trained models exist
        try:
            from asset_model import load_asset_logreg, infer_asset_logreg_from_df
            import pandas as pd

            if isinstance(ohlcv_like, pd.DataFrame):
                model, meta = load_asset_logreg()
                prob, contribs = infer_asset_logreg_from_df(ohlcv_like, model, meta)
                top = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
                assessment["risk"]["ml"] = {
                    "prob_drawdown": float(prob),
                    "top_contribs": [(k, float(v)) for k, v in top],
                    "horizon": int(meta.get("horizon", 5)),
                }
        except FileNotFoundError:
            pass
        except Exception:
            logger.exception("ML inference failed; continuing with rule-based output")

        # Attach news sentiment analysis
        try:
            from fetchers import fetch_news_articles
            from sentiment import analyze_news_sentiment

            query = base if "/" in symbol else symbol
            # Map to better search terms for NewsAPI
            query_map = {
                "LINK": "chainlink",
                "BTC": "bitcoin",
                "ETH": "ethereum",
                "ADA": "cardano",
                "DOT": "polkadot",
                "SOL": "solana",
                "AVAX": "avalanche",
                "MATIC": "polygon",
                "AAPL": "apple stock",
                "MSFT": "microsoft stock",
                "GOOGL": "google stock",
                "AMZN": "amazon stock",
                "NVDA": "nvidia stock",
            }
            query = query_map.get(query.upper(), query)  # Use base for crypto, full for stocks
            articles = fetch_news_articles(query, days=7, max_articles=20)
            try:
                news_analysis = analyze_news_sentiment(articles)
            except Exception:
                news_analysis = {"avg_sentiment": 0.0, "sentiment_volatility": 0.0, "positive_ratio": 0.0, "top_articles": [], "topics": [], "events": {}}
            assessment["news"] = {
                "sentiment": news_analysis,
                "article_count": len(articles),
                "query": query,
            }
        except Exception:
            logger.exception("News analysis failed; continuing without news")
            assessment["news"] = {"sentiment": {"avg_sentiment": 0.0}, "article_count": 0}

        # Dual model: drawdown and direction (up) probabilities
        try:
            from asset_model import load_asset_dual, infer_asset_dual_from_df
            import pandas as pd

            if isinstance(ohlcv_like, pd.DataFrame):
                dmodel, dmeta = load_asset_dual()
                p_down, p_up, c_down, c_up = infer_asset_dual_from_df(ohlcv_like, dmodel, dmeta)
                thr = dmeta.get("class_thresholds", {})
                t_down = float(thr.get("down", 0.5))
                t_up = float(thr.get("up", 0.5))
                pred_down = bool(p_down >= t_down)
                pred_up = bool(p_up >= t_up)
                topd = sorted(c_down.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
                topu = sorted(c_up.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
                assessment["risk"]["ml_dual"] = {
                    "prob_drawdown": float(p_down),
                    "prob_up": float(p_up),
                    "pred_drawdown": pred_down,
                    "pred_up": pred_up,
                    "thresholds": {"down": t_down, "up": t_up},
                    "top_contribs_down": [(k, float(v)) for k, v in topd],
                    "top_contribs_up": [(k, float(v)) for k, v in topu],
                    "horizon": int(dmeta.get("horizon", 5)),
                }
        except FileNotFoundError:
            pass
        except Exception:
            logger.exception("Dual ML inference failed; continuing with rule-based output")
        if "ml_dual" not in assessment["risk"]:
            feats = assessment.get("features", {})
            mom = float(feats.get("momentum_10", 0.0) or 0.0)
            rsi = float(feats.get("rsi_14", 50.0) or 50.0)
            vol = float(feats.get("volatility", 0.0) or 0.0)
            sent = float(assessment.get("news", {}).get("sentiment", {}).get("avg_sentiment", 0.0) or 0.0)
            adj_mom = mom
            if adj_mom > 0.3:
                adj_mom = 0.3
            if adj_mom < -0.3:
                adj_mom = -0.3
            p_up = 0.5 + adj_mom * 0.8
            if rsi >= 70.0:
                p_up -= 0.08
            elif rsi <= 30.0:
                p_up += 0.08
            p_up -= min(vol, 0.2) * 0.2
            p_up += 0.12 * sent
            if p_up < 0.01:
                p_up = 0.01
            if p_up > 0.99:
                p_up = 0.99
            p_down = 1.0 - p_up
            assessment["risk"]["ml_dual"] = {
                "prob_drawdown": float(p_down),
                "prob_up": float(p_up),
                "pred_drawdown": bool(p_down >= 0.5),
                "pred_up": bool(p_up >= 0.5),
                "thresholds": {"down": 0.5, "up": 0.5},
                "top_contribs_down": [],
                "top_contribs_up": [],
                "horizon": 5,
            }
        # Insider signals: stocks => filings; crypto => whale trades
        if "/" not in symbol:
            try:
                from fetchers import fetch_insider_transactions
                insider = fetch_insider_transactions(symbol, months=3)
                if insider:
                    assessment["insider_signals"] = insider
                else:
                    assessment["insider_signals"] = {"error": "No insider data or API not configured"}
            except Exception:
                logger.exception("Insider signals fetch failed")
                assessment["insider_signals"] = {"error": "Failed to fetch insider signals"}
        else:
            try:
                from fetchers import fetch_crypto_whale_signals
                threshold = float(os.environ.get("FEATHER_WHALE_USD", "100000"))
                whale = fetch_crypto_whale_signals(symbol, large_usd=threshold)
                if whale:
                    assessment["insider_signals"] = whale
                else:
                    assessment["insider_signals"] = {"note": "No large whale trades observed"}
            except Exception:
                logger.exception("Crypto whale signals fetch failed")
                assessment["insider_signals"] = {"error": "Failed to fetch whale signals"}
        logger.info(f"Insider signals for {symbol}: {assessment.get('insider_signals', 'NONE')}")

        return assessment
    except Exception:
        logger.exception("analyze_asset failed for %s", symbol)
        return {"features": {}, "risk": {"bucket": "green", "score": 0, "reasons": ["Analysis failed"]}}
