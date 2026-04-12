import asyncio
import os
import logging
from groq import AsyncGroq

async def generate_reasoning(dl_result: dict, ml_scores: dict, rag_data: dict) -> str:
    """
    Uses Generative AI to provide explainable reasoning. Uses Groq API if available.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    reasoning = ""
    
    try:
        client = AsyncGroq(api_key=api_key)
        
        prompt = f"""
        Analyze the following data and provide a concise, 3-sentence exact reasoning report for the user.
        
        Claim/Content: "{rag_data.get('claim')}"
        Perception Source: {dl_result.get('metadata', {}).get('source_type')}
        Deepfake Score (if video/image): {dl_result.get('metadata', {}).get('deepfake_score', 'N/A')}
        ML Credibility Score: {ml_scores.get('credibility_score')}
        ML Manipulation Risk: {ml_scores.get('manipulation_score')}
        Live Evidence Found: {rag_data.get('evidence')}
        
        Explain why it might be true, false, or manipulated based on the evidence.
        """
        
        logging.info("Generating explanation using Groq API...")
        completion = await client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are TruthLens Pro, an advanced AI fact-checker and deepfake analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        reasoning = completion.choices[0].message.content.strip()
        
    except Exception as e:
        logging.error(f"GenAI reasoning failed: {e}")
        reasoning = f"Analysis Failed: Could not generate AI reasoning. Error: {e}"

    return reasoning
