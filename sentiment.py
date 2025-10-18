import os
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

"""
Sentiment pipeline can be heavy (transformers/torch). For lightweight deployments (e.g. Render free),
set USE_VADER_ONLY=1 to skip loading transformers entirely and rely on VADER.
Optionally override HF model via HF_MODEL_NAME.
"""
MODEL_NAME = os.environ.get("HF_MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
_USE_VADER_ONLY = os.environ.get("USE_VADER_ONLY", "").strip().lower() in ("1", "true", "yes", "y")
_sentiment_pipeline = None
_vader = None

def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline
    if _USE_VADER_ONLY:
        _sentiment_pipeline = None
        return _sentiment_pipeline
    try:
        from transformers import pipeline as hf_pipeline  # lazy import
        # Use CPU on typical free hosts
        _sentiment_pipeline = hf_pipeline("sentiment-analysis", model=MODEL_NAME, device=-1)
    except Exception:
        _sentiment_pipeline = None
    return _sentiment_pipeline

def get_vader():
    global _vader
    if _vader is None:
        try:
            _vader = SentimentIntensityAnalyzer()
        except Exception:
            _vader = None
    return _vader

def lemmatize_text(text: str) -> str:
    """Lemmatize text using NLTK; fallback to simple lowercasing if NLTK is unavailable."""
    try:
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        # Tokenize and remove non-alphabetic
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha() and w not in stop_words]
        # Lemmatize
        lemmas = [lemmatizer.lemmatize(w) for w in words]
        return ' '.join(lemmas)
    except Exception:
        return text.lower()

def analyze_topics(articles: List[Dict], num_topics: int = 5) -> List[str]:
    """
    Perform improved topic modeling on news articles using LDA with lemmatization and bigrams.

    Args:
        articles: List of dicts with 'title' and 'text'.
        num_topics: Number of topics to extract.

    Returns:
        List of topic descriptions.
    """
    if not articles:
        return []

    texts = [f"{a.get('title', '')} {a.get('text', '')}".strip() for a in articles]
    texts = [t for t in texts if t]

    if len(texts) < 3:
        return ["Insufficient articles for topic modeling"]

    # Lemmatize texts (safe)
    try:
        texts = [lemmatize_text(t) for t in texts]
    except Exception:
        pass

    # Dynamic num_topics based on article count
    num_topics = min(num_topics, max(2, len(texts) // 3))

    # Try robust vectorization/LDA; fallback to top TF-IDF terms
    try:
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=3, stop_words='english', ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform(texts)
        if tfidf.shape[1] == 0:
            raise ValueError('empty vocabulary')
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(tfidf)
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
            topics.append(f"Topic {topic_idx+1}: {', '.join(top_words)}")
        return topics
    except Exception:
        # Fallback: top global TF-IDF terms
        try:
            vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english')
            tfidf = vectorizer.fit_transform(texts)
            sums = tfidf.sum(axis=0).A1
            feature_names = vectorizer.get_feature_names_out()
            top_idx = sums.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_idx]
            return [f"Topic 1: {', '.join(top_words[:5])}", f"Topic 2: {', '.join(top_words[5:10])}"]
        except Exception:
            return []

def detect_events(articles: List[Dict]) -> Dict[str, int]:
    """
    Simple event detection based on keywords.

    Args:
        articles: List of dicts.

    Returns:
        Dict of event types and counts.
    """
    event_keywords = {
        "earnings": ["earnings", "quarterly", "revenue", "profit", "loss", "guidance"],
        "merger": ["merger", "acquisition", "buyout", "takeover", "deal"],
        "regulatory": ["sec", "fda", "regulation", "lawsuit", "fine", "investigation"],
        "product": ["launch", "new product", "update", "release", "announcement"],
        "market": ["market", "volatility", "crash", "rally", "trend"],
    }

    events = {k: 0 for k in event_keywords}
    for article in articles:
        text = f"{article.get('title', '')} {article.get('text', '')}".lower()
        for event, keywords in event_keywords.items():
            if any(kw in text for kw in keywords):
                events[event] += 1

    return events

def analyze_news_sentiment(articles: List[Dict]) -> Dict:
    """
    Analyze sentiment of news articles using transformers (FinBERT preferred).
    Includes topics and events.

    Args:
        articles: List of dicts from fetch_news_articles.

    Returns:
        Dict: {"avg_sentiment": float, "sentiment_volatility": float, "positive_ratio": float, "top_articles": List[Dict], "topics": List[str], "events": Dict[str, int]}
    """
    if not articles:
        return {"avg_sentiment": 0.0, "sentiment_volatility": 0.0, "positive_ratio": 0.0, "top_articles": [], "topics": [], "events": {}}

    pipeline = get_sentiment_pipeline()
    sentiments = []

    for article in articles:
        text = f"{article.get('title', '')} {article.get('text', '')}".strip()
        score = 0.0
        if text:
            if pipeline is not None:
                try:
                    result = pipeline(text[:512])  # Limit text length
                    if isinstance(result, list) and result:
                        lab = str(result[0].get('label', '')).upper()
                        sc = float(result[0].get('score', 0.0))
                        if 'POS' in lab:
                            score = sc
                        elif 'NEG' in lab:
                            score = -sc
                except Exception:
                    pass
            if score == 0.0:
                try:
                    vader = get_vader()
                    if vader is not None:
                        vs = vader.polarity_scores(text)
                        score = float(vs.get('compound', 0.0))
                except Exception:
                    pass
        sentiments.append(score)
        article['sentiment'] = score

    import numpy as np
    sentiments = np.array(sentiments, dtype=float)
    # If all zeros, try a second pass with VADER for all texts
    if sentiments.size > 0 and np.allclose(sentiments, 0.0):
        try:
            vader = get_vader()
            if vader is not None:
                new_scores = []
                for article in articles:
                    text = f"{article.get('title', '')} {article.get('text', '')}".strip()
                    vs = vader.polarity_scores(text) if text else {"compound": 0.0}
                    s = float(vs.get('compound', 0.0))
                    article['sentiment'] = s
                    new_scores.append(s)
                sentiments = np.array(new_scores, dtype=float)
        except Exception:
            pass
    avg_sentiment = float(np.mean(sentiments)) if sentiments.size else 0.0
    sentiment_volatility = float(np.std(sentiments)) if sentiments.size else 0.0
    positive_ratio = float(np.mean(sentiments > 0)) if sentiments.size else 0.0

    top_articles = sorted(articles, key=lambda x: x.get('sentiment', 0), reverse=True)[:5]

    # Add topics and events
    # Topics/events should not crash the whole analysis
    try:
        topics = analyze_topics(articles)
    except Exception:
        topics = []
    try:
        events = detect_events(articles)
    except Exception:
        events = {}

    return {
        "avg_sentiment": float(avg_sentiment),
        "sentiment_volatility": float(sentiment_volatility),
        "positive_ratio": float(positive_ratio),
        "top_articles": top_articles,
        "topics": topics,
        "events": events,
    }
