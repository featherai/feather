from flask import Flask, render_template, request, jsonify
try:
    from flask_cors import CORS
except Exception:
    def CORS(app, *args, **kwargs):
        return app
import os
import json
import time
# Load environment variables from .env BEFORE importing modules that read them
from dotenv import load_dotenv
load_dotenv()
from risks import analyze_asset
from fetchers import news_health_check, fetch_news_articles
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    try:
        response.headers['Access-Control-Allow-Origin'] = '*'
        # Ensure caches vary by Origin to be standards-compliant
        vary = response.headers.get('Vary')
        response.headers['Vary'] = f"{vary}, Origin" if vary else 'Origin'
        response.headers['Access-Control-Allow-Credentials'] = 'false'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        # Allow common headers broadly to avoid mobile/browser quirks
        req_hdrs = request.headers.get('Access-Control-Request-Headers', '')
        allow_hdrs = ['Content-Type', 'Authorization', 'X-Requested-With', 'Accept', 'Origin']
        if req_hdrs:
            allow_hdrs.append(req_hdrs)
        response.headers['Access-Control-Allow-Headers'] = ', '.join(allow_hdrs)
        response.headers['Access-Control-Max-Age'] = '600'
    except Exception:
        pass
    return response

@app.route('/')
def index():
    return 'Feather API is running', 200

@app.route('/analyze', methods=['OPTIONS', 'POST'])
def analyze():
    if request.method == 'OPTIONS':
        return ('', 204)
    data = request.get_json()
    asset_type = data.get('type')  # crypto or stock
    asset = data.get('asset').strip()

    if not asset:
        return jsonify({'error': 'Asset required'}), 400

    # Prepare symbol
    if asset_type == 'crypto':
        symbol = asset.upper() + '/USDT'  # Assume USDT pair
    else:
        symbol = asset.upper()

    try:
        result = analyze_asset(symbol, period='1mo')
        risk = result['risk']
        ml = risk.get('ml_dual', {})
        news = result.get('news', {}).get('sentiment', {})

        coin_name = None
        base_symbol = symbol.split('/')[0]
        if len(base_symbol) > 30:
            # It's an address, fetch name
            try:
                chain = "solana"  # Assume
                url = f"https://api.coingecko.com/api/v3/coins/{chain}/contract/{base_symbol}"
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    coin_name = data.get('name', 'Token')
                else:
                    coin_name = 'Token'
            except:
                coin_name = 'Token'

        # Map common symbols to CoinGecko ids
        symbol_map = {
            'btc': 'bitcoin',
            'eth': 'ethereum',
            'ada': 'cardano',
            'bnb': 'binancecoin',
            'sol': 'solana',
            'dot': 'polkadot',
            'matic': 'matic-network',
            'avax': 'avalanche-2',
            'luna': 'terra-luna',
            'doge': 'dogecoin',
            'shib': 'shiba-inu',
            'cake': 'pancakeswap-token',
            'sushi': 'sushi',
            'uni': 'uniswap',
            'aave': 'aave',
            'comp': 'compound-governance-token',
            'link': 'chainlink',
        }
        cg_id = symbol_map.get(base_symbol.lower(), base_symbol.lower())
        try:
            from fetchers import fetch_high_quality_crypto_data
            df = fetch_high_quality_crypto_data(cg_id, days=30)
            if not df.empty:
                # Send last 100 data points
                df_sample = df.tail(100)
                price_data = [{'time': int(row['timestamp']/1000), 'value': row['price']} for _, row in df_sample.iterrows()]
            else:
                # Dummy data for testing
                price_data = [{'time': 1630000000 + i*86400, 'value': 50000 + i*100} for i in range(10)]
        except:
            # Dummy data
            price_data = [{'time': 1630000000 + i*86400, 'value': 50000 + i*100} for i in range(10)]

        response = {
            'asset': symbol,
            'coin_name': coin_name,
            'risk_level': risk['bucket'].upper(),
            'risk_score': risk['score'],
            'prob_drawdown': ml.get('prob_drawdown', 0),
            'prob_up': ml.get('prob_up', 0),
            'news_sentiment': news.get('avg_sentiment', 0),
            'article_count': result.get('news', {}).get('article_count', 0),
            'topics': news.get('topics', []),
            'events': news.get('events', {}),
            'top_articles': news.get('top_articles', []),
            'price_data': price_data
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/news_health', methods=['GET'])
def news_health():
    q = request.args.get('query', 'bitcoin')
    try:
        result = news_health_check(q, max_articles=5, timeout_sec=8)
        return jsonify({'query': q, 'providers': result})
    except Exception as e:
        return jsonify({'query': q, 'error': str(e)}), 500

@app.route('/news_fetch', methods=['GET'])
def news_fetch():
    q = request.args.get('query', 'bitcoin')
    n = int(request.args.get('n', '10'))
    try:
        arts = fetch_news_articles(q, days=7, max_articles=n)
        # only return titles and urls to keep payload small
        brief = [{"title": a.get("title"), "url": a.get("url")} for a in arts]
        return jsonify({'query': q, 'count': len(arts), 'articles': brief})
    except Exception as e:
        return jsonify({'query': q, 'error': str(e)}), 500

@app.route('/feedback', methods=['OPTIONS', 'POST'])
def feedback():
    if request.method == 'OPTIONS':
        return ('', 204)
    try:
        data = request.get_json(force=True) or {}
        comment = str(data.get('comment', '')).strip()
        if not comment:
            return jsonify({'error': 'Empty feedback'}), 400
        entry = {
            'ts': time.time(),
            'type': data.get('type'),
            'asset': data.get('asset'),
            'risk_level': data.get('risk_level'),
            'risk_score': data.get('risk_score'),
            'prob_drawdown': data.get('prob_drawdown'),
            'prob_up': data.get('prob_up'),
            'article_count': data.get('article_count'),
            'news_sentiment': data.get('news_sentiment'),
            'comment': comment[:1000],
        }
        base = os.path.dirname(__file__)
        folder = os.path.join(base, 'storage')
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, 'feedback.jsonl')
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
