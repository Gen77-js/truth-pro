import random
import asyncio
import logging
import os
from io import BytesIO

_easyocr_reader = None

# =========================
# 🖼 IMAGE MODEL (dedicated)
# haywoodsloan/ai-image-detector-deploy
# SwinV2 | ~98% accuracy on AI vs Real images
# =========================
_img_pipeline = None

def get_img_pipeline():
    global _img_pipeline
    if _img_pipeline is None:
        import torch
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        logging.info(f"Loading IMAGE deepfake model (haywoodsloan/ai-image-detector-deploy) on device {device}...")
        _img_pipeline = pipeline(
            "image-classification",
            model="haywoodsloan/ai-image-detector-deploy",
            device=device,
            framework="pt"
        )
        logging.info("Image model loaded successfully.")
    return _img_pipeline


# =========================
# 🎥 VIDEO MODEL (dedicated)
# Organika/sdxl-detector — working well for video frames
# =========================
_video_pipeline = None

def get_video_pipeline():
    global _video_pipeline
    if _video_pipeline is None:
        import torch
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        logging.info(f"Loading VIDEO deepfake model (Organika/sdxl-detector) on device {device}...")
        _video_pipeline = pipeline(
            "image-classification",
            model="Organika/sdxl-detector",
            device=device,
            framework="pt"
        )
        logging.info("Video model loaded successfully.")
    return _video_pipeline


# =========================
# 🖼 IMAGE DEEPFAKE DETECTOR
# Uses haywoodsloan/ai-image-detector-deploy
# Labels: "real" / "artificial" (or similar)
# =========================
async def _run_image_deepfake_detector(data: bytes) -> tuple:
    """
    Runs the dedicated image AI-detector model.
    Returns (fake_score: float, error_occurred: bool)
    """
    try:
        from PIL import Image
        pipe = get_img_pipeline()
        image = Image.open(BytesIO(data)).convert('RGB')

        results = await asyncio.to_thread(pipe, image)

        fake_score = 0.0
        for pred in results:
            label = str(pred.get("label", "")).lower()
            score = pred.get("score", 0.0)
            # haywoodsloan model labels: "artificial" / "ai" / "fake" = AI-generated
            if any(kw in label for kw in ["artificial", "ai", "fake", "generated"]):
                fake_score = max(fake_score, score)

        logging.info(f"[IMAGE MODEL] fake_score={fake_score:.3f} | raw={results}")
        return fake_score, False

    except Exception as e:
        logging.error(f"Image deepfake detector failed: {e}")
        return 0.5, True


# =========================
# 🎥 VIDEO FRAME DEEPFAKE DETECTOR
# Uses Organika/sdxl-detector (working well)
# =========================
async def _run_video_deepfake_detector(data: bytes) -> tuple:
    """
    Runs the dedicated video-frame AI-detector model.
    Returns (fake_score: float, error_occurred: bool)
    """
    try:
        from PIL import Image
        pipe = get_video_pipeline()
        image = Image.open(BytesIO(data)).convert('RGB')

        results = await asyncio.to_thread(pipe, image)

        fake_score = 0.0
        for pred in results:
            label = str(pred.get("label", "")).lower()
            score = pred.get("score", 0.0)
            # Organika/sdxl-detector: "artificial" = AI-generated
            if "artificial" in label:
                fake_score = max(fake_score, score)

        logging.info(f"[VIDEO MODEL] fake_score={fake_score:.3f}")
        return fake_score, False

    except Exception as e:
        logging.error(f"Video deepfake detector failed: {e}")
        return 0.5, True


# =========================
# 🔆 ERROR LEVEL ANALYSIS (ELA)
# Shared helper for both image and video
# =========================
def _run_ela_analysis(data: bytes) -> float:
    try:
        import numpy as np
        from PIL import Image, ImageChops

        quality = 90
        image = Image.open(BytesIO(data)).convert('RGB')

        # Save at lower quality in memory
        temp_buffer = BytesIO()
        image.save(temp_buffer, 'JPEG', quality=quality)
        temp_buffer.seek(0)
        compressed_image = Image.open(temp_buffer).convert('RGB')

        # Absolute difference
        diff = ImageChops.difference(image, compressed_image)

        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff

        diff = ImageChops.multiply(diff, diff.point(lambda p: p * scale))
        diff_np = np.array(diff)

        # Variance helps detect abnormal AI-smoothed surfaces or harsh spliced edges
        variance = np.var(diff_np)

        abnormal_score = 0.0
        if variance < 50:
            # AI images typically lack natural camera noise → very low variance
            abnormal_score = 0.3
        elif variance > 1000:
            # Heavily manipulated / spliced images spike above 1000
            abnormal_score = 0.3

        logging.info(f"ELA Variance: {variance:.2f} → abnormal_score: {abnormal_score}")
        return abnormal_score

    except Exception as e:
        logging.warning(f"ELA Analysis failed: {e}")
        return 0.0


# =========================
# 🖼 IMAGE PROCESSING
# =========================
async def process_image(data: bytes) -> dict:
    global _easyocr_reader

    import numpy as np
    from PIL import Image

    extracted_text = ""

    # --- OCR ---
    try:
        import easyocr
        if _easyocr_reader is None:
            _easyocr_reader = easyocr.Reader(['en'], gpu=True)

        logging.info("Running OCR...")
        image = Image.open(BytesIO(data)).convert("RGB")
        img_np = np.array(image)
        result = await asyncio.to_thread(_easyocr_reader.readtext, img_np, detail=0)
        extracted_text = " ".join(result) if result else "No text found."
    except Exception as e:
        logging.error(f"OCR failed: {e}")
        extracted_text = "OCR failed"

    # --- Dedicated Image Deepfake Detector (haywoodsloan SwinV2) ---
    logging.info("Running Image Deepfake Detection (haywoodsloan/ai-image-detector-deploy)...")
    fake_score, fake_fallback = await _run_image_deepfake_detector(data)

    # --- ELA Analysis ---
    logging.info("Running ELA Analysis...")
    abnormal_score = _run_ela_analysis(data)

    # --- Combine Scores ---
    combined_fake_score = min((fake_score + abnormal_score), 1.0)

    return {
        "content": extracted_text,
        "embeddings": [],  # embeddings handled by MiniLM later
        "metadata": {
            "source_type": "image",
            "deepfake_score": float(combined_fake_score),
            "verdict": "FAKE" if combined_fake_score > 0.6 else "REAL",
            "confidence": "High" if not fake_fallback else "Low",
            "fallback_used": fake_fallback,
            "abnormal_pattern_score": float(abnormal_score),
            "model_used": "haywoodsloan/ai-image-detector-deploy"
        }
    }


# =========================
# 🎥 VIDEO PROCESSING
# =========================
async def process_video(data: bytes) -> dict:
    import cv2
    import tempfile

    logging.info("Processing video frames (using Organika/sdxl-detector)...")

    # Save temp video file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(data)
        temp_path = tmp.name

    cap = cv2.VideoCapture(temp_path)

    frame_scores = []
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Sample every 10th frame for performance
            if frame_count % 10 == 0:
                _, buffer = cv2.imencode(".jpg", frame)
                frame_bytes = buffer.tobytes()

                score, _ = await _run_video_deepfake_detector(frame_bytes)
                ela_score = _run_ela_analysis(frame_bytes)

                combined = min(score + ela_score, 1.0)
                frame_scores.append(combined)

            frame_count += 1

    finally:
        cap.release()
        os.remove(temp_path)

    if not frame_scores:
        return {
            "content": "Video processing failed",
            "embeddings": [random.random() for _ in range(5)],
            "metadata": {
                "source_type": "video",
                "deepfake_score": 0.5,
                "verdict": "UNCERTAIN",
                "confidence": "Low",
                "fallback_used": True,
                "model_used": "Organika/sdxl-detector"
            }
        }

    final_score = sum(frame_scores) / len(frame_scores)

    return {
        "content": "Video analyzed using frame sampling locally",
        "embeddings": [random.random() for _ in range(5)],
        "metadata": {
            "source_type": "video",
            "deepfake_score": float(final_score),
            "verdict": "FAKE" if final_score > 0.6 else "REAL",
            "confidence": "High",
            "fallback_used": False,
            "model_used": "Organika/sdxl-detector"
        }
    }