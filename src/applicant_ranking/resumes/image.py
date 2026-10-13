"""Image resumes (scans/photos of resumes): re-encode through Pillow so EXIF/GPS and any
appended data are dropped. ``Image.MAX_IMAGE_PIXELS = 40_000_000``; apply
``ImageOps.exif_transpose``; ``thumbnail((2000, 2000))``; save as PNG. Result:
``ExtractionResult(kind="image", needs_vision=True, page_images=[png])`` plus a
``scanned_resume`` info flag. Unreadable image -> ``error`` + ``unreadable_file``."""

from __future__ import annotations

import io
import warnings

from PIL import Image, ImageOps

from applicant_ranking.domain import ExtractionResult, Flag, FlagCode
from applicant_ranking.resumes.extract_text import unreadable

MAX_IMAGE_PIXELS = 40_000_000
MAX_SIDE = 2000

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class ImageTooLargeError(ValueError):
    """The image declares more pixels than MAX_IMAGE_PIXELS (decompression-bomb guard)."""


def reencode_image(data: bytes) -> bytes:
    """Decode an image (first frame only) and return it as a fresh PNG, at most MAX_SIDE
    pixels on each side, orientation applied and all metadata dropped."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            source = Image.open(io.BytesIO(data))
        except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
            raise ImageTooLargeError("image declares too many pixels") from exc
        with source:
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ImageTooLargeError(f"image is {width}x{height} pixels")
            source.seek(0)
            source.draft(source.mode, (MAX_SIDE, MAX_SIDE))  # JPEG: decode at reduced scale
            source.load()
            image = ImageOps.exif_transpose(source)
    if image.mode not in ("L", "RGB"):
        if image.mode in ("1", "I", "I;16", "F"):
            image = image.convert("L")
        else:
            rgba = image.convert("RGBA")
            image = Image.new("RGB", rgba.size, (255, 255, 255))
            image.paste(rgba, mask=rgba.getchannel("A"))
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    clean = Image.new(image.mode, image.size)  # no info/EXIF carried over
    clean.paste(image)
    out = io.BytesIO()
    clean.save(out, format="PNG")
    return out.getvalue()


def extract_image(data: bytes) -> ExtractionResult:
    try:
        png = reencode_image(data)
    except ImageTooLargeError:
        return unreadable("image", "image is too large to process")
    except Exception as exc:  # Pillow raises many types on corrupt or unsupported input
        return unreadable("image", f"could not read image ({type(exc).__name__})")
    return ExtractionResult(
        kind="image", pages=1, needs_vision=True, page_images=[png],
        flags=[Flag(code=FlagCode.SCANNED_RESUME, severity="info",
                    detail="image resume: text must be transcribed from the image")],
    )
