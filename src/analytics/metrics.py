import re
import math
from typing import Dict, Any, List

def count_syllables(word: str) -> int:
    """Estimates syllable count of an English word."""
    word = word.lower()
    if len(word) <= 3:
        return 1
    word = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', word)
    word = re.sub(r'^y', '', word)
    matches = re.findall(r'[aeiouy]{1,2}', word)
    return max(1, len(matches))

def compute_flesch_reading_ease(words: List[str], sentences: List[str]) -> Dict[str, Any]:
    """Computes Flesch Reading Ease score and grade level."""
    num_words = len(words)
    num_sentences = max(1, len(sentences))
    
    if num_words == 0:
        return {"score": 0.0, "level": "N/A", "grade": "N/A"}
        
    num_syllables = sum(count_syllables(w) for w in words)
    
    # 206.835 - 1.015 * (total words / total sentences) - 84.6 * (total syllables / total words)
    score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    score = max(0.0, min(100.0, score))
    
    if score >= 90:
        level, grade = "Very Easy", "5th Grade"
    elif score >= 80:
        level, grade = "Easy", "6th Grade"
    elif score >= 70:
        level, grade = "Fairly Easy", "7th Grade"
    elif score >= 60:
        level, grade = "Standard", "8th-9th Grade"
    elif score >= 50:
        level, grade = "Fairly Difficult", "10th-12th Grade (High School)"
    elif score >= 30:
        level, grade = "Difficult", "College / University"
    else:
        level, grade = "Very Difficult", "Academic / Professional"
        
    return {
        "score": round(score, 1),
        "level": level,
        "grade": grade
    }

def compute_document_analytics(text: str, filename: str = "document") -> Dict[str, Any]:
    """Computes rich reading metrics and linguistic statistics for a document."""
    if not text:
        return {
            "filename": filename,
            "word_count": 0,
            "char_count": 0,
            "sentence_count": 0,
            "paragraph_count": 0,
            "reading_time_min": 0,
            "speaking_time_min": 0,
            "vocabulary_richness": 0.0,
            "unique_words": 0,
            "readability": {"score": 0.0, "level": "N/A", "grade": "N/A"}
        }

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    # Sentence splitting
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    # Word tokenization
    words = re.findall(r'\b\w+\b', text.lower())
    
    word_count = len(words)
    char_count = len(text)
    sentence_count = max(1, len(sentences))
    paragraph_count = max(1, len(paragraphs))
    
    # Reading metrics (avg reading speed = 220 wpm, speaking = 140 wpm)
    reading_time_min = round(word_count / 220, 1)
    speaking_time_min = round(word_count / 140, 1)
    
    unique_words = len(set(words))
    # Type-token ratio
    ttr = round((unique_words / word_count) * 100, 1) if word_count > 0 else 0.0
    
    # Readability
    readability = compute_flesch_reading_ease(words, sentences)
    
    avg_words_per_sentence = round(word_count / sentence_count, 1)
    avg_chars_per_word = round(sum(len(w) for w in words) / max(1, word_count), 1)

    return {
        "filename": filename,
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "reading_time_min": max(0.1, reading_time_min),
        "speaking_time_min": max(0.1, speaking_time_min),
        "vocabulary_richness_pct": ttr,
        "unique_words": unique_words,
        "avg_words_per_sentence": avg_words_per_sentence,
        "avg_chars_per_word": avg_chars_per_word,
        "readability": readability
    }
