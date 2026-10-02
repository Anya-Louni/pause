"""Create the sample inputs used by the notebook: an event flyer and a voice note.

Both are synthetic. The flyer is drawn with Pillow and the voice note is spoken
by the operating system's offline voice.
"""

from pathlib import Path

import pyttsx3
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "samples"


def flyer() -> None:
    image = Image.new("RGB", (900, 620), "#fff6dc")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 870, 590), outline="#1b1b1b", width=6)
    draw.text((70, 80), "DESIGN MEETUP", fill="#1b1b1b", font=ImageFont.load_default(size=84))
    lines = [
        "Thursday 15 October",
        "6:00 pm to 8:00 pm",
        "Room 4, Mill Road Library",
        "Free. Bring a sketchbook.",
    ]
    for i, line in enumerate(lines):
        draw.text((70, 230 + i * 80), line, fill="#1b1b1b", font=ImageFont.load_default(size=52))
    image.save(OUT / "flyer.png")


def voice_note() -> None:
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.save_to_file(
        "Add a task to renew my passport, due next Monday.", str(OUT / "voice_note.wav")
    )
    engine.runAndWait()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    flyer()
    voice_note()
    print(sorted(p.name for p in OUT.iterdir()))
