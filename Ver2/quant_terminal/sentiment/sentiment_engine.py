"""
sentiment/sentiment_engine.py
Market Sentiment Engine — Quant Research Terminal

Features:
  - VADER sentiment scoring on news headlines
  - NewsAPI integration for live headlines
  - Rolling sentiment aggregate
  - Sentiment-to-signal conversion
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



def _load_env_file() -> None:
    """Load key=value pairs from .env into os.environ (non-overwriting)."""
    # Walk up from this file's location to find quant_terminal root
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in [here, os.path.join(here, ".."), os.path.join(here, "..", "..")]:
        env_path = os.path.join(folder, ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
            break  # stop at first .env found


_load_env_file()

NEWS_API_KEY = os.environ.get("722f109a8ca2440ca9511b1201bc17dc", "")
NEWS_API_URL = "https://newsapi.org/v2/everything"
# ──────────────────────────────────────────────

_analyzer = SentimentIntensityAnalyzer()


def score_text(text: str) -> dict:
    """
    Score a single headline/text using VADER.

    Returns dict:
      compound : -1.0 to +1.0 (main signal)
      positive : 0 to 1
      negative : 0 to 1
      neutral  : 0 to 1
      label    : "Bullish" / "Bearish" / "Neutral"
    """
    scores  = _analyzer.polarity_scores(str(text))
    compound = scores["compound"]

    label = "Neutral"
    if compound >= 0.05:
        label = "Bullish"
    elif compound <= -0.05:
        label = "Bearish"

    return {
        "compound" : round(compound, 4),
        "positive" : round(scores["pos"], 4),
        "negative" : round(scores["neg"], 4),
        "neutral"  : round(scores["neu"], 4),
        "label"    : label
    }


def score_headlines(headlines: list) -> pd.DataFrame:
    """
    Score a list of headline strings.
    Returns DataFrame with sentiment scores.
    """
    rows = []
    for h in headlines:
        row = score_text(h)
        row["headline"] = h
        rows.append(row)
    return pd.DataFrame(rows)[["headline", "compound", "positive", "negative", "neutral", "label"]]



def fetch_news(
    query: str,
    days_back: int  = 7,
    max_articles: int = 50,
    language: str   = "en"
) -> pd.DataFrame:
    """
    Fetch recent news from NewsAPI for a given query.

    Parameters
    ----------
    query        : Search term e.g. "Apple AAPL stock"
    days_back    : How many days of news to retrieve
    max_articles : Maximum article count (free tier: 100/day)

    Returns
    -------
    DataFrame with: title, description, publishedAt, source, url
    """
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "q"        : query,
        "from"     : from_date,
        "sortBy"   : "publishedAt",
        "language" : language,
        "pageSize" : min(max_articles, 100),
        "apiKey"   : NEWS_API_KEY
    }

    if not NEWS_API_KEY:
        print("[NewsAPI] ERROR: NEWS_API_KEY not found. Create a .env file in quant_terminal/ with: NEWS_API_KEY=your_key")
        return pd.DataFrame()
    print(f"[NewsAPI] Key prefix : {NEWS_API_KEY[:8]}...")
    print(f"[NewsAPI] Query      : {query} | from: {from_date}")

    try:
        resp = requests.get(NEWS_API_URL, params=params, timeout=10)
        print(f"[NewsAPI] Status     : {resp.status_code}")
        data = resp.json()

        if data.get("status") != "ok":
            print(f"[NewsAPI] Error      : {data.get('message', 'Unknown error')}")
            return pd.DataFrame()

        articles = data.get("articles", [])
        print(f"[NewsAPI] Articles   : {len(articles)}")
        rows = []
        for a in articles:
            rows.append({
                "title"       : a.get("title", ""),
                "description" : a.get("description", ""),
                "published_at": a.get("publishedAt", ""),
                "source"      : a.get("source", {}).get("name", ""),
                "url"         : a.get("url", "")
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
            df.sort_values("published_at", ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df

    except Exception as e:
        print(f"[NewsAPI] Request failed: {e}")
        return pd.DataFrame()



def run_sentiment_pipeline(
    symbol: str,
    company_name: str = None,
    days_back: int    = 14,
    max_articles: int = 50
) -> dict:
    """
    Full pipeline: fetch news → score → aggregate.

    Returns dict with:
      articles_df      : Raw articles with sentiment scores
      aggregate        : Overall sentiment summary
      daily_sentiment  : Daily rolling sentiment (for chart overlay)
    """
    query = f"{company_name or symbol} {symbol} stock"

    articles = fetch_news(query, days_back=days_back, max_articles=max_articles)

    if articles.empty:
        return {
            "articles_df"    : pd.DataFrame(),
            "aggregate"      : {"compound": 0.0, "label": "No Data", "n": 0},
            "daily_sentiment": pd.Series(dtype=float)
        }

    # Score each headline
    articles["combined_text"] = articles["title"].fillna("") + " " + articles["description"].fillna("")
    scores = score_headlines(articles["combined_text"].tolist())

    articles = pd.concat([
        articles.reset_index(drop=True),
        scores[["compound", "positive", "negative", "neutral", "label"]]
    ], axis=1)

    # Aggregate
    mean_compound = articles["compound"].mean()
    agg_label     = "Neutral"
    if mean_compound >= 0.05:
        agg_label = "Bullish"
    elif mean_compound <= -0.05:
        agg_label = "Bearish"

    aggregate = {
        "compound"     : round(mean_compound, 4),
        "label"        : agg_label,
        "n"            : len(articles),
        "bullish_pct"  : round((articles["label"] == "Bullish").mean() * 100, 1),
        "bearish_pct"  : round((articles["label"] == "Bearish").mean() * 100, 1),
        "neutral_pct"  : round((articles["label"] == "Neutral").mean() * 100, 1),
    }

    # Daily sentiment bullshit
    articles["date"] = articles["published_at"].dt.date
    daily_sentiment  = articles.groupby("date")["compound"].mean()
    daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
    daily_sentiment = daily_sentiment.sort_index()

    # Rolling 3-day smoothed meri gand me dal do
    daily_sentiment_smooth = daily_sentiment.rolling(3, min_periods=1).mean()

    return {
        "articles_df"          : articles,
        "aggregate"            : aggregate,
        "daily_sentiment"      : daily_sentiment,
        "daily_sentiment_smooth": daily_sentiment_smooth
    }


#demo le lo
def demo_sentiment(n: int = 30) -> pd.Series:
    """
    Generate synthetic sentiment data for demo/testing without API key.
    Returns a time-indexed Series of sentiment scores with EXACTLY n entries.
    Uses n+10 periods then slices [-n:] to guarantee correct length regardless
    of weekends/holidays in pd.date_range with freq="B".
    """
    np.random.seed(42)
    dates  = pd.date_range(end=datetime.today(), periods=n + 10, freq="B")[-n:]
    scores = np.random.normal(0.05, 0.25, len(dates)).clip(-1, 1)
    return pd.Series(scores.round(4), index=dates, name="sentiment")


#Qucikei
if __name__ == "__main__":
    print("=== VADER Scoring Test ===")
    headlines = [
        "Apple hits record high as iPhone sales surge globally",
        "Federal Reserve signals more rate hikes amid inflation concerns",
        "Tech stocks rally on strong earnings season",
        "Market crash fears grow as volatility spikes",
        "Trading volume unchanged during midday session"
    ]
    df = score_headlines(headlines)
    print(df.to_string(index=False))

    print("\n=== Demo Sentiment Series ===")
    demo = demo_sentiment()
    print(demo.tail(5))