# Vendored from claimcheck (github.com/Anya-Louni/claimcheck) at commit 4dca7b7.
# Kept as a copy so this repo runs on its own.

"""Validate uploaded images and audio and turn them into content blocks."""

import base64
import io
from typing import Any

from PIL import Image, ImageOps

IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
AUDIO_TYPES = {"audio/wav", "audio/x-wav", "audio/webm", "audio/ogg", "audio/mpeg", "audio/mp4"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_IMAGE_SIDE = 1600


class MediaError(ValueError):
    """Raised when an upload fails validation."""


def clean_image(data: bytes, mime_type: str) -> bytes:
    """Check type and size, drop all metadata (EXIF and GPS), downscale, return PNG bytes."""
    if mime_type not in IMAGE_TYPES:
        raise MediaError(f"unsupported image type {mime_type}")
    if len(data) > MAX_IMAGE_BYTES:
        raise MediaError("image is larger than 8 MB")
    try:
        image: Image.Image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:
        raise MediaError("file is not a readable image") from exc
    image = ImageOps.exif_transpose(image)
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    # Copying pixels into a fresh image leaves every metadata chunk behind.
    fresh = Image.frombytes(image.mode, image.size, image.tobytes())
    out = io.BytesIO()
    fresh.save(out, format="PNG")
    return out.getvalue()


def check_audio(data: bytes, mime_type: str) -> None:
    """Check audio type and size."""
    base_type = mime_type.split(";")[0].strip()
    if base_type not in AUDIO_TYPES:
        raise MediaError(f"unsupported audio type {mime_type}")
    if len(data) > MAX_AUDIO_BYTES:
        raise MediaError("audio is larger than 10 MB")


def image_block(png: bytes) -> dict[str, Any]:
    """Return a standard LangChain image content block."""
    return {"type": "image", "base64": base64.b64encode(png).decode(), "mime_type": "image/png"}


def audio_block(data: bytes, mime_type: str) -> dict[str, Any]:
    """Return a standard LangChain audio content block."""
    base_type = mime_type.split(";")[0].strip()
    return {"type": "audio", "base64": base64.b64encode(data).decode(), "mime_type": base_type}
