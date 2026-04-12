from dl_layer.vision import process_image, process_video
from dl_layer.audio import process_audio
from dl_layer.text import process_text

async def route_dl_processing(content_type: str, data: bytes = None, text_data: str = None) -> dict:
    """
    Routes the input to the appropriate perception model.
    """
    if content_type == "text":
        return await process_text(text_data)
    elif content_type == "image":
        return await process_image(data)
    elif content_type == "video":
        return await process_video(data)
    elif content_type == "audio":
        return await process_audio(data)
    else:
        return {"content": "Unknown document content", "embeddings": [], "metadata": {"source_type": "document"}}
