import re
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class DocumentChunk:
    chunk_id: str
    page_number: int
    content: str
    word_count: int
    char_count: int
    metadata: Dict[str, Any]

def split_text_recursive(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    separators: List[str] = None
) -> List[str]:
    """Recursively splits text on natural boundaries (paragraphs, sentences, words)."""
    if not text:
        return []
        
    if separators is None:
        separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
        
    chunks = []
    
    # Try splitting by first separator
    def _split(current_text: str, sep_idx: int) -> List[str]:
        if len(current_text) <= chunk_size or sep_idx >= len(separators):
            return [current_text] if current_text.strip() else []
            
        sep = separators[sep_idx]
        splits = current_text.split(sep)
        result = []
        acc = ""
        
        for s in splits:
            candidate = (acc + sep + s) if acc else s
            if len(candidate) <= chunk_size:
                acc = candidate
            else:
                if acc:
                    result.append(acc)
                if len(s) > chunk_size:
                    # Recursive split on next granular separator
                    sub_splits = _split(s, sep_idx + 1)
                    result.extend(sub_splits)
                    acc = ""
                else:
                    acc = s
        if acc:
            result.append(acc)
        return result

    raw_chunks = _split(text, 0)
    
    # Apply overlap window
    if chunk_overlap > 0 and len(raw_chunks) > 1:
        merged = []
        for i, c in enumerate(raw_chunks):
            if i > 0:
                # Prepend the tail of previous chunk as overlap context
                overlap_text = raw_chunks[i-1][-chunk_overlap:]
                c_with_overlap = f"... {overlap_text.strip()} {c}"
                merged.append(c_with_overlap)
            else:
                merged.append(c)
        return merged
        
    return raw_chunks

def split_document_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[DocumentChunk]:
    """Splits each parsed page into chunks preserving page citations and metadata."""
    chunks: List[DocumentChunk] = []
    global_chunk_idx = 0
    
    for page in pages:
        page_num = page.get("page_number", 1)
        page_text = page.get("cleaned_text") or page.get("raw_text") or ""
        
        if not page_text.strip():
            continue
            
        page_splits = split_text_recursive(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        for p_idx, text_chunk in enumerate(page_splits, start=1):
            global_chunk_idx += 1
            chunks.append(DocumentChunk(
                chunk_id=f"p{page_num}_c{p_idx}",
                page_number=page_num,
                content=text_chunk.strip(),
                word_count=len(text_chunk.split()),
                char_count=len(text_chunk),
                metadata={
                    "page": page_num,
                    "index_in_page": p_idx,
                    "global_index": global_chunk_idx
                }
            ))
            
    return chunks
