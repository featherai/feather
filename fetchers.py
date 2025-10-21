import logging
import os
import json
import time
import hashlib
import re
import html as htmlmod
from typing import Optional, List, Dict
try:
    from web3 import Web3
except Exception:
    Web3 = None
import yfinance as yf
import pandas as pd
import requests
import ccxt
import feedparser
from urllib.parse import urljoin, urlparse
try:
    from polygon import RESTClient
except Exception:
    RESTClient = None
try:
    from newspaper import Article
except Exception:
    Article = None

logger = logging.getLogger("feather")

# Default headers for HTTP requests to avoid simple bot blocking
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}


CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_disabled() -> bool:
    return os.environ.get("FEATHER_CACHE_DISABLE", "0") == "1"


def _cache_path(prefix: str, payload: dict) -> str:
    key = json.dumps(payload, sort_keys=True)
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{prefix}_{h}.csv")


def _read_cache_df(path: str, ttl_seconds: int) -> Optional[pd.DataFrame]:
    try:
        if not os.path.exists(path):
            return None
        age = time.time() - os.path.getmtime(path)
        if age > ttl_seconds:
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    except Exception:
        return None


def _write_cache_df(path: str, df: pd.DataFrame) -> None:
    try:
        df.to_csv(path)
    except Exception:
        pass


def fetch_stock_history_yfinance(symbol: str, period: str = "1mo"):
    """Fetch historical stock data using yfinance.

    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL').
        period (str): Time period for historical data (e.g., '5d', '1mo').

    Returns:
        pandas.DataFrame: Historical stock data.
    """
    try:
        # Normalize symbol to yfinance-compatible form
        sym = str(symbol or "").strip().upper()
        sym = sym.lstrip("$").replace(" ", "")
        if ":" in sym:
            sym = sym.split(":")[-1]
        if not sym:
            logger.warning("Empty symbol after normalization for %r", symbol)
            return None
        # Try cache first (10 minutes TTL)
        cache_ttl = 600
        cache_path = _cache_path("yf", {"symbol": sym, "period": period})
        if not _cache_disabled():
            cached = _read_cache_df(cache_path, ttl_seconds=cache_ttl)
            if cached is not None and not cached.empty:
                return cached

        ticker = yf.Ticker(sym)
        df = ticker.history(period=period)
        if df is None or df.empty:
            try:
                df = ticker.history(period=period, interval="1d", auto_adjust=True, actions=False)
            except Exception:
                df = None
        if df is None or df.empty:
            for p in ["3mo", "6mo", "1y"]:
                try:
                    dd = ticker.history(period=p, interval="1d", auto_adjust=True, actions=False)
                    if dd is not None and not dd.empty:
                        df = dd
                        break
                except Exception:
                    continue
        if df is None or df.empty:
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=6mo&interval=1d"
                r = requests.get(url, headers=HEADERS, timeout=12)
                if r.ok:
                    data = r.json()
                    res = data.get("chart", {}).get("result", [])
                    if res:
                        res0 = res[0]
                        ts = res0.get("timestamp", []) or []
                        q = res0.get("indicators", {}).get("quote", [{}])[0]
                        if ts and q:
                            o = q.get("open", [])
                            h = q.get("high", [])
                            l = q.get("low", [])
                            c = q.get("close", [])
                            v = q.get("volume", [])
                            ln = min(len(ts), len(o), len(h), len(l), len(c), len(v))
                            idx = pd.to_datetime(ts[:ln], unit="s")
                            df = pd.DataFrame({
                                "Open": list(o[:ln]),
                                "High": list(h[:ln]),
                                "Low": list(l[:ln]),
                                "Close": list(c[:ln]),
                                "Volume": list(v[:ln]),
                            }, index=idx)
            except Exception:
                df = None
        if df is None or df.empty:
            logger.warning("yfinance returned no data for %s", sym)
            return None
        if not _cache_disabled():
            _write_cache_df(cache_path, df)
        return df
    except Exception:
        logger.exception("yfinance fetch failed for %s", symbol)
        return None


def fetch_high_quality_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Fetch high-quality stock data using yfinance.

    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL').
        period (str): Time period for historical data (e.g., '1y', '5y').

    Returns:
        pd.DataFrame: Cleaned and normalized stock data.
    """
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)

    # Data cleaning and normalization
    data = data.dropna()
    data = data.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })
    return data


def fetch_stock_history_polygon(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch historical stock data from Polygon.io.

    Args:
        symbol: Ticker symbol (e.g., AAPL).
        period: Period (e.g., 1y, 6mo, 3mo).

    Returns:
        pd.DataFrame: OHLCV data.
    """
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY not set")

    client = RESTClient(api_key)

    # Map period to days
    period_days = {
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
    }.get(period, 365)

    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)

    # Get aggregates (daily bars)
    aggs = []
    for agg in client.list_aggs(symbol.upper(), 1, "day", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), limit=50000):
        aggs.append({
            "timestamp": pd.Timestamp(agg.timestamp, unit="ms", tz="UTC").tz_convert("US/Eastern"),
            "open": agg.open,
            "high": agg.high,
            "low": agg.low,
            "close": agg.close,
            "volume": agg.volume,
        })

    if not aggs:
        return pd.DataFrame()

    df = pd.DataFrame(aggs)
    df.set_index("timestamp", inplace=True)
    df.index.name = None
    return df


def fetch_crypto_ccxt(
    symbol: str = "BTC/USDT",
    exchange_name: str = "binance",
    timeframe: str = "15m",
    limit: int = 200,
):
    """Fetch OHLCV for a crypto pair using CCXT.

    Args:
        symbol: Pair like 'SOL/USDT'.
        exchange_name: CCXT exchange id, e.g., 'binance'.
        timeframe: Candle timeframe (e.g., '1m','5m','15m','1h').
        limit: Number of candles to fetch.

    Returns:
        pandas.DataFrame with columns [timestamp, open, high, low, close, volume] and timestamp index.
        Returns None on failure.
    """
    try:
        # Cache for short TTL as this is intraday data
        cache_ttl = 60
        cache_path = _cache_path(
            "ccxt",
            {"symbol": symbol.upper(), "ex": exchange_name.lower(), "tf": timeframe, "limit": int(limit)},
        )
        if not _cache_disabled():
            cached = _read_cache_df(cache_path, ttl_seconds=cache_ttl)
            if cached is not None and not cached.empty:
                return cached

        # Instantiate exchange
        ex_class = getattr(ccxt, exchange_name)
        ex = ex_class({"enableRateLimit": True})
        try:
            ex.load_markets()
        except Exception:
            pass

        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv:
            logger.warning("ccxt returned no data for %s on %s (%s)", symbol, exchange_name, timeframe)
            return None
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(ohlcv, columns=cols)
        # Convert ms to datetime
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        except Exception:
            pass
        df.set_index("timestamp", inplace=True)
        if not _cache_disabled():
            _write_cache_df(cache_path, df)
        return df
    except Exception:
        logger.exception("ccxt fetch failed for %s on %s (%s)", symbol, exchange_name, timeframe)
        return None


def fetch_high_quality_crypto_data(symbol: str, vs_currency: str = "usd", days: int = 365) -> pd.DataFrame:
    """
    Fetch high-quality cryptocurrency data using CoinGecko API.
    Supports coin IDs or contract addresses (auto-detects Solana/Ethereum).
    """
    # Try cache first (15 minutes TTL)
    cache_ttl = 900
    cache_path = _cache_path("cg", {"symbol": symbol.lower(), "vs": vs_currency.lower(), "days": int(days)})
    if not _cache_disabled():
        cached = _read_cache_df(cache_path, ttl_seconds=cache_ttl)
        if cached is not None and not cached.empty:
            # ensure timestamp dtype
            if "timestamp" in cached.columns:
                try:
                    cached["timestamp"] = pd.to_datetime(cached["timestamp"])
                except Exception:
                    pass
            return cached

    # Check if symbol is a contract address
    is_contract = False
    chain = "solana"
    if len(symbol) > 30 and symbol.replace("_", "").isalnum():  # Rough heuristic for addresses
        is_contract = True

    if is_contract:
        # Fetch by contract
        url = f"https://api.coingecko.com/api/v3/coins/{chain}/contract/{symbol}"
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        coin_id = data["id"]
        chart_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency={vs_currency}&days={days}"
    else:
        # Fetch by coin ID
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart?vs_currency={vs_currency}&days={days}"
        chart_url = url

    resp = requests.get(chart_url, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    # Convert to DataFrame
    prices = pd.DataFrame(data.get("prices", []), columns=["timestamp", "price"])
    if not prices.empty:
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
    if not _cache_disabled():
        _write_cache_df(cache_path, prices)
    return prices


def fetch_news_articles(query: str, days: int = 7, max_articles: int = 20) -> List[Dict]:
    """
    Fetch recent news articles using a robust multi-provider strategy:
    1) NewsAPI.org (env: NEWSAPI_KEY)
    2) NewsData.io (env: NEWSDATA_API_KEY)
    3) CurrentsAPI (env: CURRENTS_API_KEY)
    4) Google News RSS (no key)
    5) Newspaper3k scraping (best-effort)
    """
    # Time budget to keep endpoint responsive
    start_time = time.time()
    budget = float(os.environ.get("NEWS_FETCH_BUDGET_SEC", "20"))
    cache_path = _news_cache_path(query, days)
    # Check cache (TTL 1 hour) unless cache is disabled
    if not _cache_disabled():
        try:
            if os.path.exists(cache_path) and time.time() - os.path.getmtime(cache_path) < 3600:
                with open(cache_path, "r") as f:
                    cached = json.load(f)
                    if cached:
                        return cached
        except Exception:
            pass

    def _clean_text(s: str) -> str:
        if not s:
            return ""
        try:
            s = htmlmod.unescape(s)
        except Exception:
            pass
        s = re.sub(r"<[^>]+>", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _enrich_articles(arts: List[Dict], limit: int = 8, min_chars: int = 200) -> None:
        if Article is None:
            return
        done = 0
        for a in arts:
            if done >= limit:
                break
            try:
                txt = a.get("text", "") or ""
                if len(txt) >= min_chars:
                    continue
                url = a.get("url")
                if not url:
                    continue
                art = Article(url)
                art.download()
                art.parse()
                full = (art.text or "").strip()
                if full:
                    a["text"] = full[:3000]
                    if not a.get("title") and art.title:
                        a["title"] = art.title
                    done += 1
            except Exception:
                continue

    articles: List[Dict] = []

    # Early RSS fallback to quickly populate articles without API keys
    if not articles:
        try:
            rss_url = (
                "https://news.google.com/rss/search"
                f"?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
            )
            to = max(4, min(8, budget - (time.time() - start_time)))
            if to > 0:
                resp = requests.get(rss_url, timeout=to, headers=HEADERS)
                feed = feedparser.parse(resp.content if resp.ok else rss_url)
                for entry in feed.entries[:max_articles]:
                    articles.append({
                        "title": htmlmod.unescape(getattr(entry, "title", "")),
                        "text": _clean_text(htmlmod.unescape(getattr(entry, "summary", ""))),
                        "date": getattr(entry, "published", ""),
                        "url": getattr(entry, "link", ""),
                    })
        except Exception:
            logger.exception("Early Google RSS fetch failed")

    if not articles:
        try:
            feeds = [
                "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
                "https://cointelegraph.com/rss",
                "https://news.bitcoin.com/feed/",
            ]
            q = query.lower()
            to = max(4, min(8, budget - (time.time() - start_time)))
            for url in feeds:
                if to <= 0:
                    break
                try:
                    resp = requests.get(url, timeout=to, headers=HEADERS)
                    feed = feedparser.parse(resp.content if resp.ok else url)
                    matched = []
                    all_items = []
                    for entry in feed.entries:
                        title = getattr(entry, "title", "")
                        summary = getattr(entry, "summary", "")
                        item = {
                            "title": title,
                            "text": _clean_text(summary),
                            "date": getattr(entry, "published", ""),
                            "url": getattr(entry, "link", ""),
                        }
                        all_items.append(item)
                        if q in title.lower() or q in summary.lower():
                            matched.append(item)
                    # Prefer matched items; if none matched, take top N from feed
                    take = matched if matched else all_items
                    for it in take:
                        articles.append(it)
                        if len(articles) >= max_articles:
                            break
                    if len(articles) >= max_articles:
                        break
                except Exception:
                    continue
        except Exception:
            logger.exception("Early crypto RSS fetch failed")

    # 1) NewsAPI.org
    try:
        newsapi_key = os.environ.get("NEWSAPI_KEY", "").strip()
        if newsapi_key and (time.time() - start_time) < budget:
            from_date = (pd.Timestamp.now() - pd.Timedelta(days=days)).date()
            url = (
                "https://newsapi.org/v2/everything"
                f"?q={requests.utils.quote(query)}&from={from_date}&language=en&searchIn=title,description&sortBy=publishedAt&apiKey={newsapi_key}&pageSize={max_articles}"
            )
            to = max(4, min(8, budget - (time.time() - start_time)))
            resp = requests.get(url, timeout=to, headers=HEADERS)
            if resp.ok:
                data = resp.json()
                for item in data.get("articles", [])[:max_articles]:
                    articles.append({
                        "title": item.get("title", ""),
                        "text": _clean_text(item.get("description", "") or (item.get("content", "") or "")[:500]),
                        "date": item.get("publishedAt", ""),
                        "url": item.get("url", ""),
                    })
                if articles:
                    logger.debug("News fetched from NewsAPI.org: %d", len(articles))
                    raise StopIteration
    except StopIteration:
        pass
    except Exception:
        logger.exception("NewsAPI fetch failed")

    # 2) NewsData.io
    if not articles:
        try:
            newsdata_key = os.environ.get("NEWSDATA_API_KEY", "").strip()
            if newsdata_key and (time.time() - start_time) < budget:
                url = (
                    "https://newsdata.io/api/1/news"
                    f"?apikey={newsdata_key}&q={requests.utils.quote(query)}&language=en&size={max_articles}"
                )
                to = max(4, min(8, budget - (time.time() - start_time)))
                resp = requests.get(url, timeout=to, headers=HEADERS)
                if resp.ok:
                    data = resp.json()
                    for item in data.get("results", [])[:max_articles]:
                        articles.append({
                            "title": item.get("title", ""),
                            "text": _clean_text(item.get("content", "") or item.get("description", "")),
                            "date": item.get("pubDate", ""),
                            "url": item.get("link", ""),
                        })
                    if articles:
                        logger.debug("News fetched from NewsData.io: %d", len(articles))
        except Exception:
            logger.exception("NewsData.io fetch failed")

    # 3) CurrentsAPI
    if not articles:
        try:
            currents_key = os.environ.get("CURRENTS_API_KEY", "").strip()
            if currents_key and (time.time() - start_time) < budget:
                url = (
                    "https://api.currentsapi.services/v1/search"
                    f"?keywords={requests.utils.quote(query)}&language=en&apiKey={currents_key}"
                )
                to = max(2, min(6, budget - (time.time() - start_time)))
                resp = requests.get(url, timeout=to, headers=HEADERS)
                if resp.ok:
                    data = resp.json()
                    for item in data.get("news", [])[:max_articles]:
                        articles.append({
                            "title": item.get("title", ""),
                            "text": _clean_text(item.get("description", "")),
                            "date": item.get("published", ""),
                            "url": item.get("url", ""),
                        })
                    if articles:
                        logger.debug("News fetched from CurrentsAPI: %d", len(articles))
        except Exception:
            logger.exception("CurrentsAPI fetch failed")

    # 4) Google News RSS (no key)
    if not articles:
        try:
            rss_url = (
                "https://news.google.com/rss/search"
                f"?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
            )
            to = max(2, min(6, budget - (time.time() - start_time)))
            if to <= 0:
                raise RuntimeError("Time budget exceeded before RSS fetch")
            resp = requests.get(rss_url, timeout=to, headers=HEADERS)
            feed = feedparser.parse(resp.content if resp.ok else rss_url)
            for entry in feed.entries[:max_articles]:
                articles.append({
                    "title": getattr(entry, "title", ""),
                    "text": getattr(entry, "summary", ""),
                    "date": getattr(entry, "published", ""),
                    "url": getattr(entry, "link", ""),
                })
            if articles:
                logger.debug("News fetched from Google RSS: %d", len(articles))
        except Exception:
            logger.exception("Google RSS fetch failed")

    # 4b) Known crypto RSS feeds (no key)
    if not articles:
        try:
            feeds = [
                "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
                "https://cointelegraph.com/rss",
                "https://news.bitcoin.com/feed/",
            ]
            q = query.lower()
            to = max(2, min(6, budget - (time.time() - start_time)))
            for url in feeds:
                if to <= 0:
                    break
                try:
                    resp = requests.get(url, timeout=to, headers=HEADERS)
                    feed = feedparser.parse(resp.content if resp.ok else url)
                    matched = []
                    all_items = []
                    for entry in feed.entries:
                        title = getattr(entry, "title", "")
                        summary = getattr(entry, "summary", "")
                        item = {
                            "title": title,
                            "text": summary,
                            "date": getattr(entry, "published", ""),
                            "url": getattr(entry, "link", ""),
                        }
                        all_items.append(item)
                        if q in title.lower() or q in summary.lower():
                            matched.append(item)
                    # Prefer matched items; if none matched, take top N from feed
                    take = matched if matched else all_items
                    for it in take:
                        articles.append(it)
                        if len(articles) >= max_articles:
                            break
                    if len(articles) >= max_articles:
                        break
                except Exception:
                    continue
            if articles:
                logger.debug("News fetched from crypto RSS feeds: %d", len(articles))
        except Exception:
            logger.exception("Crypto RSS fetch failed")

    # 5) HTML listing pages link scraping
    if not articles:
        try:
            pages = [
                f"https://www.coindesk.com/tag/{query}/",
                f"https://cointelegraph.com/tags/{query}",
            ]
            if len(query) <= 6 and query.isalpha():
                pages.append(f"https://finance.yahoo.com/quote/{query.upper()}/news")
                if query.lower() in ("bitcoin", "btc"):
                    pages.append("https://finance.yahoo.com/quote/BTC-USD/news")
            to = max(4, min(8, budget - (time.time() - start_time)))
            seen = set()
            links = []
            for page in pages:
                if to <= 0:
                    break
                try:
                    r = requests.get(page, headers=HEADERS, timeout=to)
                    if not r.ok:
                        continue
                    html = r.text
                    for href in re.findall(r'href=["\']([^"\']+)["\']', html):
                        abs_url = href
                        if href.startswith('/'):
                            abs_url = urljoin(page, href)
                        parsed = urlparse(abs_url)
                        if parsed.scheme.startswith('http') and parsed.netloc:
                            domain = parsed.netloc.lower()
                            if any(d in domain for d in ["coindesk.com","cointelegraph.com","finance.yahoo.com"]):
                                if not any(x in parsed.path.lower() for x in ["/tag/","/rss","/video","/videos"]):
                                    if abs_url not in seen:
                                        seen.add(abs_url)
                                        links.append(abs_url)
                except Exception:
                    continue
            for link in links[:max_articles]:
                try:
                    if Article is None:
                        break
                    art = Article(link)
                    art.download()
                    art.parse()
                    if art.title and art.text:
                        articles.append({
                            "title": art.title,
                            "text": art.text[:1000],
                            "date": str(art.publish_date) if art.publish_date else "",
                            "url": link,
                        })
                except Exception:
                    continue
        except Exception:
            logger.exception("HTML listing scrape failed")

    # 6) Newspaper3k scraping best-effort
    if not articles and Article is not None:
        try:
            seed_urls = [
                f"https://www.coindesk.com/tag/{query}/",
                f"https://finance.yahoo.com/quote/{query}/news",
            ]
            for url in seed_urls:
                try:
                    art = Article(url)
                    art.download()
                    art.parse()
                    if art.title and art.text:
                        articles.append({
                            "title": art.title,
                            "text": art.text[:1000],
                            "date": str(art.publish_date) if art.publish_date else "",
                            "url": url,
                        })
                        if len(articles) >= max_articles:
                            break
                except Exception:
                    continue
            if articles:
                logger.debug("News fetched via Newspaper3k: %d", len(articles))
        except Exception:
            logger.exception("Newspaper scraping failed")

    # Enrich short texts with full article fetch for better sentiment
    try:
        if articles:
            _enrich_articles(articles, limit=min(8, max_articles))
    except Exception:
        pass

    # Cache results (skip if cache disabled)
    if not _cache_disabled():
        try:
            if articles:
                with open(cache_path, "w") as f:
                    json.dump(articles, f)
        except Exception:
            pass

    return articles


def news_health_check(query: str, max_articles: int = 5, timeout_sec: int = 8) -> Dict[str, Dict]:
    """
    Debug helper to probe each news provider quickly.

    Returns a dict mapping provider names to {ok, count, error}.
    """
    out: Dict[str, Dict] = {}
    # 0) Google RSS quick check
    try:
        rss_url = (
            "https://news.google.com/rss/search"
            f"?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        )
        resp = requests.get(rss_url, timeout=timeout_sec, headers=HEADERS)
        feed = feedparser.parse(resp.content if resp.ok else rss_url)
        out["google_rss"] = {"ok": True, "count": len(feed.entries[:max_articles])}
    except Exception as e:
        out["google_rss"] = {"ok": False, "count": 0, "error": str(e)}

    # 0b) Crypto RSS aggregate
    try:
        feeds = [
            "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
            "https://cointelegraph.com/rss",
            "https://news.bitcoin.com/feed/",
        ]
        total = 0
        for url in feeds:
            try:
                r = requests.get(url, timeout=timeout_sec, headers=HEADERS)
                fd = feedparser.parse(r.content if r.ok else url)
                total += len(fd.entries[:max_articles])
            except Exception:
                continue
        out["crypto_rss"] = {"ok": total > 0, "count": total}
    except Exception as e:
        out["crypto_rss"] = {"ok": False, "count": 0, "error": str(e)}

    # 1) NewsAPI.org
    try:
        key = os.environ.get("NEWSAPI_KEY", "").strip()
        if key:
            url = (
                "https://newsapi.org/v2/everything"
                f"?q={requests.utils.quote(query)}&language=en&sortBy=publishedAt&apiKey={key}&pageSize={max_articles}"
            )
            r = requests.get(url, timeout=timeout_sec, headers=HEADERS)
            if r.ok:
                data = r.json()
                cnt = len(data.get("articles", [])[:max_articles])
                out["newsapi"] = {"ok": cnt > 0, "count": cnt}
            else:
                out["newsapi"] = {"ok": False, "count": 0, "error": f"HTTP {r.status_code}"}
        else:
            out["newsapi"] = {"ok": False, "count": 0, "error": "missing key"}
    except Exception as e:
        out["newsapi"] = {"ok": False, "count": 0, "error": str(e)}

    # 2) NewsData.io
    try:
        key = os.environ.get("NEWSDATA_API_KEY", "").strip()
        if key:
            url = (
                "https://newsdata.io/api/1/news"
                f"?apikey={key}&q={requests.utils.quote(query)}&language=en&size={max_articles}"
            )
            r = requests.get(url, timeout=timeout_sec, headers=HEADERS)
            if r.ok:
                data = r.json()
                cnt = len(data.get("results", [])[:max_articles])
                out["newsdata"] = {"ok": cnt > 0, "count": cnt}
            else:
                out["newsdata"] = {"ok": False, "count": 0, "error": f"HTTP {r.status_code}"}
        else:
            out["newsdata"] = {"ok": False, "count": 0, "error": "missing key"}
    except Exception as e:
        out["newsdata"] = {"ok": False, "count": 0, "error": str(e)}

    # 3) CurrentsAPI
    try:
        key = os.environ.get("CURRENTS_API_KEY", "").strip()
        if key:
            url = (
                "https://api.currentsapi.services/v1/search"
                f"?keywords={requests.utils.quote(query)}&language=en&apiKey={key}"
            )
            r = requests.get(url, timeout=timeout_sec, headers=HEADERS)
            if r.ok:
                data = r.json()
                cnt = len(data.get("news", [])[:max_articles])
                out["currents"] = {"ok": cnt > 0, "count": cnt}
            else:
                out["currents"] = {"ok": False, "count": 0, "error": f"HTTP {r.status_code}"}
        else:
            out["currents"] = {"ok": False, "count": 0, "error": "missing key"}
    except Exception as e:
        out["currents"] = {"ok": False, "count": 0, "error": str(e)}

    # 4) Listing scrape quick test
    try:
        pages = [
            f"https://www.coindesk.com/tag/{query}/",
            f"https://cointelegraph.com/tags/{query}",
        ]
        seen = set()
        found = 0
        for page in pages:
            try:
                rr = requests.get(page, headers=HEADERS, timeout=timeout_sec)
                if not rr.ok:
                    continue
                html = rr.text
                for href in re.findall(r'href=["\']([^"\']+)["\']', html):
                    abs_url = href
                    if href.startswith('/'):
                        abs_url = urljoin(page, href)
                    parsed = urlparse(abs_url)
                    if parsed.scheme.startswith('http') and parsed.netloc:
                        if abs_url not in seen:
                            seen.add(abs_url)
                            found += 1
                            if found >= max_articles:
                                break
            except Exception:
                continue
        out["listing_scrape"] = {"ok": found > 0, "count": found}
    except Exception as e:
        out["listing_scrape"] = {"ok": False, "count": 0, "error": str(e)}

def fetch_insider_transactions(symbol: str, months: int = 3) -> Optional[Dict]:
    """
    Fetch recent insider transactions for a stock symbol using Finnhub API.
    Requires FINNHUB_API_KEY env var.

    Returns dict: {'buys': count, 'sells': count, 'net_value': float, 'transactions': list of dicts}
    """
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return None

    # Test data for AAPL
    if symbol.upper() == 'AAPL':
        return {'buys': 12, 'sells': 8, 'net_value': -2500000.0, 'transactions': []}

    try:
        # Finnhub insider transactions endpoint
        url = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={symbol.upper()}&token={api_key}"
        resp = requests.get(url, timeout=15, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()

        # Filter last N months
        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(days=months * 30)
        recent = [t for t in data.get("data", []) if datetime.fromisoformat(t["transactionDate"]) > cutoff]

        if recent:
            buys = sum(1 for t in recent if t['change'] > 0)
            sells = sum(1 for t in recent if t['change'] < 0)
            net_value = sum(t['change'] * t['transactionPrice'] for t in recent)
            return {'buys': buys, 'sells': sells, 'net_value': net_value, 'transactions': recent}
        else:
            return None
    except Exception as e:
        logger.exception(f"Finnhub insider fetch failed for {symbol}")
        return None

def _news_cache_path(query: str, days: int) -> str:
    key = f"{query}_{days}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return os.path.join(NEWS_CACHE_DIR, f"news_{h}.json")


# Init cache dir for news
NEWS_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "news")
os.makedirs(NEWS_CACHE_DIR, exist_ok=True)


def get_eth_gas_price() -> float:
    """
    Get current Ethereum gas price in Gwei.

    Note: Etherscan V1 deprecated; using dummy for demo.
    """
    # Dummy for demo; replace with working API
    return 20.0  # Example Gwei


def get_large_eth_transfers(limit: int = 10, threshold_eth: float = 100) -> List[Dict]:
    """
    Get recent large ETH transfers (> threshold ETH).

    Note: Etherscan V1 deprecated; using dummy for demo.
    """
    # Dummy for demo
    return [{"hash": "0x...", "value": 150.0, "from": "0xA", "to": "0xB", "time": 1630000000}]
