from dl_layer.router import route_dl_processing
from ml_layer.scorer import score_content
from rag.retriever import retrieve_evidence
from genai.reasoning import generate_reasoning
import time

async def process_content(content_type: str, data: bytes = None, filename: str = None) -> dict:
    start_time = time.time()
    
    # 1. Perception Layer (DL)
    dl_result = await route_dl_processing(content_type, data, text_data=data if content_type=="text" else None)
    
    # 2. Scoring Layer (ML)
    ml_scores = score_content(dl_result)
    
    # 3. Verification Layer (RAG)
    rag_data = await retrieve_evidence(dl_result)
    
    # 4. Reasoning Layer (GenAI)
    reasoning = await generate_reasoning(dl_result, ml_scores, rag_data)
    
    elapsed = time.time() - start_time
    
    return {
        "status": "success",
        "processing_time": f"{elapsed:.2f}s",
        "input_metadata": {
            "type": content_type,
            "filename": filename
        },
        "perception": dl_result,
        "scores": ml_scores,
        "evidence": rag_data,
        "explanation": reasoning
    }
