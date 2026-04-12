import asyncio
import logging

_embedding_model = None

async def retrieve_evidence(dl_result: dict) -> dict:
    """
    Retrieves real-time evidence using DuckDuckGo and Wikipedia with fallbacks.
    """
    text = dl_result.get("content", "")
    claim = text[:50] + "..." if len(text) > 50 else text
    evidence = []
    
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        import wikipedia
        
        logging.info("Performing live web search for evidence...")
        
        # 1. Search DuckDuckGo News or Web
        # Just searching the first few words as keywords
        keywords = " ".join(text.split()[:10]).strip()
        
        if not keywords:
            logging.info("No text found in input to trigger Web check. Falling back to base response.")
            return {
                "claim": "No explicit text claims detected in media.",
                "evidence": ["No text extracted from upload to verify against live sources."]
            }
        
        # We run synchronous libraries in thread
        def perform_search():
            results = []
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(keywords, max_results=2))
                for r in ddg_results:
                    results.append(f"Web: {r.get('body', '')}")
            try:
                wiki_summary = wikipedia.summary(keywords, sentences=1)
                results.append(f"Wikipedia: {wiki_summary}")
            except:
                pass
            return results

        real_evidence = await asyncio.to_thread(perform_search)
        
        if real_evidence:
            evidence.extend(real_evidence)
        else:
            evidence.append("No direct evidence found on the live web for this claim.")
            
    except Exception as e:
        logging.error(f"RAG search failed: {e}")
        evidence.append(f"Search error occurred: {e}")
    
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logging.info("Loading MiniLM model...")
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            logging.warning("SentenceTransformers not installed, skipping MiniLM embeddings.")
            
    embeddings = []
    if _embedding_model is not None and text and len(text) > 2:
        try:
            embeddings_np = await asyncio.to_thread(_embedding_model.encode, [text])
            embeddings = embeddings_np[0].tolist()
        except:
            pass

    # Update original dict so pipeline has access to actual embeddings
    dl_result["embeddings"] = embeddings

    return {
        "claim": claim,
        "evidence": evidence
    }
