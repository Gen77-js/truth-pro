import logging
import os
import asyncio

# Using Inference API to save RAM for the hackathon
HF_API_KEY = os.getenv("HF_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
}

def fetch_hf_text(model_id, payload):
    import requests
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    response = requests.post(url, headers=HEADERS, json=payload)
    try:
        return response.json()
    except Exception:
        return None

async def process_text(text: str) -> dict:
    source = "text"
    
    # Defaults in case API fails
    contradiction_score = 0.5
    sentiment_score = 0.5 # 0 is negative, 1 is positive
    
    # 1. BART for Contradiction Detection / Claim Validation
    try:
        logging.info("Running BART for contradiction...")
        bart_res = await asyncio.to_thread(
            fetch_hf_text, 
            "facebook/bart-large-mnli", 
            {"inputs": text, "parameters": {"candidate_labels": ["truth", "lie", "contradiction", "exaggeration"]}}
        )
        
        if isinstance(bart_res, dict) and "labels" in bart_res:
            labels = bart_res["labels"]
            scores = bart_res["scores"]
            
            for lbl, sc in zip(labels, scores):
                if lbl in ["lie", "contradiction"]:
                    contradiction_score = max(contradiction_score, sc)
                if lbl == "exaggeration":
                    # Boost sentiment toward negative if exaggerated
                    sentiment_score = min(sentiment_score, 1 - sc)
    except Exception as e:
        logging.error(f"BART text inference failed: {e}")

    # 2. DistilBERT for Sentiment / Exaggeration
    try:
        logging.info("Running DistilBERT for sentiment...")
        distil_res = await asyncio.to_thread(
            fetch_hf_text,
            "distilbert-base-uncased-finetuned-sst-2-english",
            {"inputs": text}
        )
        if isinstance(distil_res, list) and len(distil_res) > 0 and isinstance(distil_res[0], list):
            for res in distil_res[0]:
                if res["label"] == "NEGATIVE":
                    # Negative sentiment might correlate with fake panic news
                    sentiment_score = 1.0 - res["score"]
                elif res["label"] == "POSITIVE":
                    sentiment_score = res["score"]
    except Exception as e:
        logging.error(f"DistilBERT inference failed: {e}")

    return {
        "content": text,
        "embeddings": [], # will be added by retriever
        "metadata": {
            "source_type": source,
            "contradiction_score": float(contradiction_score),
            "sentiment_score": float(sentiment_score)
        }
    }
