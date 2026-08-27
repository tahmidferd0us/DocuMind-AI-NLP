import nltk
from typing import List, Optional

# Ensure required NLTK resources
def ensure_nltk_data():
    for resource in ["punkt", "punkt_tab", "stopwords"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except (LookupError, IndexError):
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass

def generate_extractive_summary(text: str, sentence_count: int = 5, method: str = "lexrank") -> dict:
    """
    Generates an extractive summary by ranking and selecting the most central sentences.
    Methods: 'lexrank' (graph centrality), 'lsa' (latent semantic analysis), 'textrank'.
    100% offline, zero API quota usage.
    """
    if not text or not text.strip():
        return {
            "summary": "",
            "sentences": [],
            "sentence_count": 0,
            "method": method
        }
        
    ensure_nltk_data()
    
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words
    
    LANGUAGE = "english"
    stemmer = Stemmer(LANGUAGE)
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    
    if method == "lsa":
        summarizer = LsaSummarizer(stemmer)
    elif method == "textrank":
        summarizer = TextRankSummarizer(stemmer)
    else:
        summarizer = LexRankSummarizer(stemmer)
        
    summarizer.stop_words = get_stop_words(LANGUAGE)
    
    extracted = summarizer(parser.document, sentence_count)
    selected_sentences = [str(s).strip() for s in extracted if str(s).strip()]
    
    # Fallback to simple first sentences if sumy returns empty on short text
    if not selected_sentences:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        selected_sentences = sentences[:sentence_count]
        
    summary_text = " ".join(selected_sentences)
    
    return {
        "summary": summary_text,
        "sentences": selected_sentences,
        "sentence_count": len(selected_sentences),
        "method": method
    }
