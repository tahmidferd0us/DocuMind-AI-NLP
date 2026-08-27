from typing import Dict, Any, Optional
from ..config import generate_with_retry, is_gemini_configured, logger

ABSTRACTIVE_SYSTEM_INSTRUCTION = """You are an expert academic and technical document summariser for DocuMind AI.
Your task is to produce a precise, high-fidelity abstractive summary of the provided text.
Guidelines:
1. Synthesize key arguments, findings, methodologies, and conclusions into clear, coherent prose.
2. Maintain factual accuracy strictly faithful to the source text without hallucinating facts.
3. Eliminate redundant fluff and conversational filler.
"""

def generate_abstractive_summary(
    text: str,
    format_type: str = "paragraph",
    max_length: str = "standard",
    focus_topic: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a neural abstractive summary using Gemini Flash with rate-limit protection.
    format_type: 'paragraph', 'bullet_points', 'executive'
    max_length: 'brief' (100 words), 'standard' (250 words), 'detailed' (500 words)
    """
    if not text or not text.strip():
        return {
            "summary": "",
            "format_type": format_type,
            "max_length": max_length,
            "model_used": "none"
        }

    if not is_gemini_configured():
        return {
            "summary": "Gemini API key is not configured. Please set GEMINI_API_KEY in documind-nlp/.env to enable neural abstractive summarisation.",
            "format_type": format_type,
            "max_length": max_length,
            "model_used": "unconfigured"
        }

    length_guidelines = {
        "brief": "Target length: 100-150 words. Focus strictly on the primary objective and conclusion.",
        "standard": "Target length: 200-300 words. Cover the problem, approach, key evidence, and outcome.",
        "detailed": "Target length: 450-600 words. Comprehensive breakdown including methodology, nuanced points, and future scope."
    }

    format_guidelines = {
        "paragraph": "Write in well-structured paragraphs with clear narrative flow.",
        "bullet_points": "Format the response as bullet points under clear category headers (e.g. • Key Objectives, • Findings, • Implications).",
        "executive": "Provide an Executive Summary with an Overview paragraph followed by 4-6 high-impact takeaway bullet points."
    }

    prompt = f"""Summarise the following document according to these specifications:
- {length_guidelines.get(max_length, length_guidelines['standard'])}
- {format_guidelines.get(format_type, format_guidelines['paragraph'])}
{f"- Special Focus: Prioritise information relevant to '{focus_topic}'." if focus_topic else ""}

DOCUMENT TEXT:
---
{text[:40000]}
---
"""

    try:
        summary_result = generate_with_retry(
            prompt=prompt,
            system_instruction=ABSTRACTIVE_SYSTEM_INSTRUCTION
        )
        return {
            "summary": summary_result.strip(),
            "format_type": format_type,
            "max_length": max_length,
            "model_used": "gemini-flash"
        }
    except Exception as e:
        logger.error(f"Abstractive summary generation failed: {e}")
        return {
            "summary": f"Failed to generate abstractive summary: {str(e)}",
            "format_type": format_type,
            "max_length": max_length,
            "model_used": "error"
        }
