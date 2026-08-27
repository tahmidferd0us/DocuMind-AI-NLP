import os
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env from current directory or documind-nlp root
base_dir = Path(__file__).resolve().parent.parent
load_dotenv(base_dir / ".env")
load_dotenv()

logger = logging.getLogger("documind_nlp")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")

_client = None

def get_gemini_client():
    """Returns a singleton instance of the google-genai Client."""
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please add GEMINI_API_KEY=your_key to documind-nlp/.env"
            )
        from google import genai
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client

def is_gemini_configured() -> bool:
    return bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here")

def generate_with_retry(prompt: str, system_instruction: str = None, model: str = None, max_retries: int = 3) -> str:
    """Executes Gemini generation with exponential backoff to handle Free Tier rate limits."""
    client = get_gemini_client()
    target_model = model or GEMINI_MODEL
    
    config = {}
    if system_instruction:
        config["system_instruction"] = system_instruction
        
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=target_model,
                contents=prompt,
                config=config if config else None
            )
            return response.text or ""
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "rate" in err_str:
                wait_time = (2 ** attempt) * 2 + 1
                logger.warning(f"Rate limited (attempt {attempt + 1}/{max_retries}). Backing off for {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Gemini generation error: {e}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2)
    raise RuntimeError("Failed to generate content after maximum retries.")
