import random
import asyncio
import tempfile
import os
import logging

# We will try to lazy load Whisper to save memory and catch installation errors
_whisper_model = None

async def process_audio(data: bytes) -> dict:
    global _whisper_model
    
    extracted_text = ""
    source = "audio"
    
    try:
        # Write bytes to a temp file for Whisper
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        import whisper
        # Load the small model if not already loaded
        if _whisper_model is None:
            logging.info("Loading Whisper 'small' model for the first time...")
            _whisper_model = whisper.load_model("small")
            
        logging.info("Transcribing audio via Whisper...")
        # run in thread so we don't block asyncio
        result = await asyncio.to_thread(_whisper_model.transcribe, tmp_path)
        extracted_text = result.get("text", "").strip()
        
        # Cleanup
        os.remove(tmp_path)
        
    except Exception as e:
        logging.warning(f"Whisper inference failed, falling back to simulation. Error: {e}")
        # Simulated Whisper Speech-to-Text Fallback
        await asyncio.sleep(1)
        extracted_text = "This is a transcribed audio text claiming something sensational."
    
    
    # Audio Analysis Logic
    suspicious_audio_score = 0.0
    text_lower = extracted_text.lower()
    
    # Unnatural pauses / stutters which often happen in poorly generated audio
    stutter_count = text_lower.count(" uh ") + text_lower.count(" um ") + text_lower.count("...") + text_lower.count(" err ")
    if stutter_count > 3:
        suspicious_audio_score += 0.2
        
    weird_phrasing = ["i mean", "like", "you know"] # Often lacking in pure AI speech, or overly used in bad clones
    for phrase in weird_phrasing:
        if text_lower.count(phrase) > 2:
            suspicious_audio_score += 0.1

    suspicious_audio_score = min(suspicious_audio_score, 1.0)

    return {
        "content": extracted_text,
        "embeddings": [], # Embeddings handled by MiniLM
        "metadata": {
            "source_type": source, 
            "fallback_used": extracted_text == "This is a transcribed audio text claiming something sensational.",
            "suspicious_audio_score": suspicious_audio_score
        }
    }
