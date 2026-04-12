def score_content(dl_result: dict) -> dict:
    """
    Evaluates credibility and risk based on calculated perception outputs!
    """
    meta = dl_result.get("metadata", {})
    source = meta.get("source_type", "text")
    
    cred_score = 0.8
    bias_score = 0.2
    manip_score = 0.1
    
    if source == "text":
        contradict = meta.get("contradiction_score", 0.0)
        sentiment = meta.get("sentiment_score", 0.5)
        
        manip_score += contradict * 0.5
        cred_score -= contradict * 0.6
        
        if sentiment < 0.3:
            bias_score += 0.4 # highly negative could imply exaggeration/bias
    elif source in ["image", "video"]:
        df_score = meta.get("deepfake_score", 0.0)
        abnorm = meta.get("abnormal_pattern_score", 0.0)
        manip_score = df_score + abnorm
        cred_score -= (df_score * 0.8)
    elif source == "audio":
        sus_audio = meta.get("suspicious_audio_score", 0.0)
        manip_score += sus_audio * 0.8
        cred_score -= sus_audio * 0.6
        
    # Bounds bounds
    cred_score = max(0.01, min(cred_score, 0.99))
    bias_score = max(0.01, min(bias_score, 0.99))
    manip_score = max(0.01, min(manip_score, 0.99))
        
    return {
        "credibility_score": round(cred_score, 2),
        "bias_score": round(bias_score, 2),
        "manipulation_score": round(manip_score, 2)
    }
