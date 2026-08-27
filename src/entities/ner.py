import spacy
from collections import Counter
from typing import Dict, List, Any
from ..config import logger

_nlp = None

def get_spacy_nlp():
    """Loads or downloads the lightweight en_core_web_sm model."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.info("Downloading spaCy model 'en_core_web_sm'...")
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp

def extract_entities(text: str, max_text_chars: int = 100000) -> Dict[str, Any]:
    """
    Extracts and aggregates named entities using spaCy.
    100% offline, zero API quota.
    Returns grouped entities with frequencies and unique counts.
    """
    if not text or not text.strip():
        return {
            "entities_by_type": {},
            "top_entities": [],
            "total_entities_found": 0
        }

    nlp = get_spacy_nlp()
    
    # Process text in chunks if very long to prevent spaCy memory limit
    doc = nlp(text[:max_text_chars])
    
    label_map = {
        "PERSON": "People",
        "ORG": "Organizations",
        "GPE": "Locations & Geopolitics",
        "LOC": "Locations",
        "DATE": "Dates & Times",
        "MONEY": "Financial Amounts",
        "EVENT": "Events",
        "PRODUCT": "Products & Systems",
        "LAW": "Laws & Regulations",
        "NORP": "Nationalities / Groups"
    }

    entities_grouped: Dict[str, Counter] = {}
    all_entities = Counter()

    for ent in doc.ents:
        clean_name = ent.text.strip().replace("\n", " ")
        if len(clean_name) <= 1 or clean_name.isdigit():
            continue
            
        group_name = label_map.get(ent.label_, ent.label_)
        if group_name not in entities_grouped:
            entities_grouped[group_name] = Counter()
            
        entities_grouped[group_name][clean_name] += 1
        all_entities[clean_name] += 1

    formatted_groups = {
        group: [{"name": name, "count": count} for name, count in counter.most_common(15)]
        for group, counter in entities_grouped.items()
    }

    top_entities = [
        {"name": name, "count": count} for name, count in all_entities.most_common(20)
    ]

    return {
        "entities_by_type": formatted_groups,
        "top_entities": top_entities,
        "total_entities_found": sum(all_entities.values())
    }
