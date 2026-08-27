from typing import List, Dict, Any, Optional
from .vector_store import VectorStore
from ..config import generate_with_retry, is_gemini_configured, logger

RAG_SYSTEM_INSTRUCTION = """You are a precision document Question Answering assistant for DocuMind AI.
Your answers MUST be strictly grounded in the provided document passages.
Rules:
1. Rely ONLY on the facts stated directly in the CONTEXT below. Do NOT assume, extrapolate, or bring outside facts.
2. For every key statement you make, cite the page number using bracket format, e.g., [Page 2].
3. If the provided context does not contain sufficient information to answer the question, explicitly state:
   "Based on the provided document, there is not enough information to answer this question."
4. Be concise, objective, and clear.
"""

class RAGEngine:
    """Manages context retrieval and grounded generative Q&A with Gemini."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def answer_question(
        self,
        question: str,
        top_k: int = 4,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Retrieves relevant passages and generates a grounded response with source citations."""
        if not question or not question.strip():
            return {
                "question": "",
                "answer": "Please enter a valid question.",
                "sources": [],
                "model_used": "none"
            }

        # 1. Retrieve top passages
        search_results = self.vector_store.search(question, top_k=top_k)
        
        if not search_results:
            return {
                "question": question,
                "answer": "No document content has been indexed yet. Please upload a document first.",
                "sources": [],
                "model_used": "none"
            }

        # Format sources and context
        context_blocks = []
        sources = []
        
        for idx, (chunk, score) in enumerate(search_results, start=1):
            context_blocks.append(f"--- [Passage {idx} | Page {chunk.page_number}] ---\n{chunk.content}")
            sources.append({
                "page": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "similarity_score": round(score, 3),
                "snippet": chunk.content[:250] + ("..." if len(chunk.content) > 250 else "")
            })

        context_str = "\n\n".join(context_blocks)

        # Build prompt
        history_str = ""
        if chat_history:
            turns = []
            for turn in chat_history[-3:]: # Keep last 3 turns
                role = "User" if turn.get("role") == "user" else "Assistant"
                turns.append(f"{role}: {turn.get('content', '')}")
            if turns:
                history_str = "PREVIOUS CONVERSATION:\n" + "\n".join(turns) + "\n\n"

        prompt = f"""{history_str}CONTEXT PASSAGES FROM DOCUMENT:
{context_str}

QUESTION:
{question}

Provide your grounded answer with citations [Page X]:"""

        if not is_gemini_configured():
            # Graceful offline mode: return retrieved passages directly
            fallback_answer = (
                "⚠️ **Gemini API key is not configured.** Showing the most relevant passages retrieved from the document:\n\n" +
                "\n\n".join([f"**[Page {s['page']}]** {s['snippet']}" for s in sources])
            )
            return {
                "question": question,
                "answer": fallback_answer,
                "sources": sources,
                "model_used": "offline-retrieval"
            }

        try:
            generated_answer = generate_with_retry(
                prompt=prompt,
                system_instruction=RAG_SYSTEM_INSTRUCTION
            )
            return {
                "question": question,
                "answer": generated_answer.strip(),
                "sources": sources,
                "model_used": "gemini-flash"
            }
        except Exception as e:
            logger.error(f"RAG generation failed: {e}")
            return {
                "question": question,
                "answer": f"Error generating answer: {str(e)}",
                "sources": sources,
                "model_used": "error"
            }
