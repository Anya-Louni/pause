import io

import pytest
from PIL import Image

from pause.media import MediaError, check_audio, clean_image, image_block


def _jpeg_with_gps() -> bytes:
    image = Image.new("RGB", (3000, 2000), "red")
    exif = Image.Exif()
    exif[0x8825] = {1: "N", 2: (48.0, 51.0, 24.0)}  # GPSInfo
    exif[0x010F] = "PhoneMaker"
    out = io.BytesIO()
    image.save(out, format="JPEG", exif=exif)
    return out.getvalue()


def test_clean_image_strips_exif_and_downscales() -> None:
    raw = _jpeg_with_gps()
    assert Image.open(io.BytesIO(raw)).getexif()
    cleaned = Image.open(io.BytesIO(clean_image(raw, "image/jpeg")))
    assert not cleaned.getexif()
    assert not cleaned.info.get("exif")
    assert max(cleaned.size) == 1600


def test_rejects_wrong_type_and_garbage() -> None:
    with pytest.raises(MediaError):
        clean_image(b"x", "image/gif")
    with pytest.raises(MediaError):
        clean_image(b"not an image", "image/png")


def test_audio_checks() -> None:
    check_audio(b"x" * 10, "audio/webm;codecs=opus")
    with pytest.raises(MediaError):
        check_audio(b"x", "video/mp4")
    with pytest.raises(MediaError):
        check_audio(b"x" * (11 * 1024 * 1024), "audio/wav")


def test_image_block_shape() -> None:
    block = image_block(b"\x89PNG")
    assert block["type"] == "image" and block["mime_type"] == "image/png"
