"""Готовим изменённые копии снимков для проверки поиска дубликатов."""

import io

from PIL import Image, ImageDraw


def variants(data):
    source = Image.open(io.BytesIO(data)).convert("RGB")
    result = {}

    def save(name, image, format="PNG", **kwargs):
        target = io.BytesIO()
        image.save(target, format=format, **kwargs)
        result[name] = target.getvalue()

    save("jpeg", source, "JPEG", quality=45)
    save("resize", source.resize((max(32, source.width // 2), max(32, source.height // 2))))
    w, h = source.size
    save("crop", source.crop((int(w * 0.12), int(h * 0.12), int(w * 0.88), int(h * 0.88))))
    overlay = Image.new("RGBA", source.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((w * 0.1, h * 0.7, w * 0.9, h * 0.85), fill=(255, 255, 255, 140))
    draw.text(
        (w * 0.15, h * 0.73),
        "SAMPLE WATERMARK",
        fill=(30, 30, 30, 190),
        font_size=max(12, int(w * 0.045)),
    )
    save("watermark", Image.alpha_composite(source.convert("RGBA"), overlay).convert("RGB"))
    return result
