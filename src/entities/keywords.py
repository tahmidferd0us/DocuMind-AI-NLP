from typing import List, Dict, Any, Tuple
from ..config import logger

_kw_model = None

def get_keybert_model():
    """Initializes KeyBERT with the shared SentenceTransformer instance."""
    global _kw_model
    if _kw_model is None:
        try:
            from keybert import KeyBERT
            from ..rag.vector_store import get_embedder
            embedder = get_embedder()
            _kw_model = KeyBERT(model=embedder)
        except Exception as e:
            logger.warning(f"KeyBERT initialization note: {e}")
            _kw_model = None
    return _kw_model

def extract_keywords_tfidf(text: str, top_n: int = 15) -> List[Dict[str, Any]]:
    """Fallback keyword extractor using frequency / TF-IDF."""
    import re
    from collections import Counter
    from nltk.corpus import stopwords
    
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = {"the", "and", "is", "in", "to", "of", "a", "for", "on", "with", "as", "by", "at", "an", "this", "that"}

    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered = [w for w in words if w not in stop_words]
    counts = Counter(filtered)
    
    total = sum(counts.values()) or 1
    return [{"keyword": word, "score": round(count / total * 10, 3)} for word, count in counts.most_common(top_n)]

def extract_keywords(
    text: str,
    top_n: int = 15,
    keyphrase_ngram_range: Tuple[int, int] = (1, 2)
) -> List[Dict[str, Any]]:
    """
    Extracts key phrases using KeyBERT with MMR for fast, diverse topic ranking.
    100% offline, zero API quota.
    """
    if not text or not text.strip():
        return []

    sample_text = text[:8000]
    
    try:
        kw_model = get_keybert_model()
        if kw_model is not None:
            keywords = kw_model.extract_keywords(
                sample_text,
                keyphrase_ngram_range=keyphrase_ngram_range,
                stop_words="english",
                use_mmr=True,
                diversity=0.4,
                top_n=top_n
            )
            return [{"keyword": kw, "score": round(float(score), 3)} for kw, score in keywords]
    except Exception as e:
        logger.warning(f"KeyBERT failed, falling back to frequency/TF-IDF: {e}")

    return extract_keywords_tfidf(sample_text, top_n=top_n)
