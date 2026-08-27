from typing import Dict, Any, List, Union
from rouge_score import rouge_scorer
import sacrebleu

def evaluate_summary(reference: str, candidate: str) -> Dict[str, Any]:
    """
    Computes ROUGE-1, ROUGE-2, ROUGE-L, and SacreBLEU scores between reference and generated candidate.
    """
    if not reference.strip() or not candidate.strip():
        return {
            "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            "bleu": {"score": 0.0}
        }

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)

    # BLEU
    try:
        bleu_res = sacrebleu.corpus_bleu([candidate], [[reference]])
        bleu_score = round(bleu_res.score, 2)
    except Exception:
        bleu_score = 0.0

    return {
        "rouge1": {
            "precision": round(scores['rouge1'].precision * 100, 2),
            "recall": round(scores['rouge1'].recall * 100, 2),
            "fmeasure": round(scores['rouge1'].fmeasure * 100, 2)
        },
        "rouge2": {
            "precision": round(scores['rouge2'].precision * 100, 2),
            "recall": round(scores['rouge2'].recall * 100, 2),
            "fmeasure": round(scores['rouge2'].fmeasure * 100, 2)
        },
        "rougeL": {
            "precision": round(scores['rougeL'].precision * 100, 2),
            "recall": round(scores['rougeL'].recall * 100, 2),
            "fmeasure": round(scores['rougeL'].fmeasure * 100, 2)
        },
        "bleu": {
            "score": bleu_score
        }
    }

def evaluate_dataset(pairs: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Evaluates an entire dataset of (reference, candidate) summary pairs and computes aggregate averages.
    """
    if not pairs:
        return {}

    r1_f, r2_f, rl_f, bleus = [], [], [], []

    for item in pairs:
        res = evaluate_summary(item["reference"], item["candidate"])
        r1_f.append(res["rouge1"]["fmeasure"])
        r2_f.append(res["rouge2"]["fmeasure"])
        rl_f.append(res["rougeL"]["fmeasure"])
        bleus.append(res["bleu"]["score"])

    n = len(pairs)
    return {
        "total_evaluated": n,
        "avg_rouge1_fmeasure": round(sum(r1_f) / n, 2),
        "avg_rouge2_fmeasure": round(sum(r2_f) / n, 2),
        "avg_rougeL_fmeasure": round(sum(rl_f) / n, 2),
        "avg_bleu_score": round(sum(bleus) / n, 2)
    }
