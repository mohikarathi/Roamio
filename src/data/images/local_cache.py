"""Local image cache and Base64 data URI resolver for 100% reliable offline rendering."""

import base64
import hashlib
from pathlib import Path
from typing import Optional

CACHE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "images_cache"

STATIC_IMAGES_DIR = Path(__file__).parent.parent.parent.parent / "static" / "images"

_DATA_URI_CACHE = {}


def get_local_image_path(url: Optional[str]) -> Optional[Path]:
    """Return local file path for image URL if cached on disk."""
    if not url:
        return None
    url_hash = hashlib.md5(url.encode()).hexdigest()
    candidate = CACHE_DIR / f"{url_hash}.jpg"
    if candidate.exists():
        return candidate
    candidate_static = STATIC_IMAGES_DIR / f"{url_hash}.jpg"
    if candidate_static.exists():
        return candidate_static
    return None


def get_image_web_url(url: Optional[str]) -> str:
    """Return local static web path (/app/static/images/{hash}.jpg) for zero-latency browser rendering."""
    if not url:
        return ""
    local_path = get_local_image_path(url)
    if local_path and local_path.exists():
        # Ensure file exists in static dir
        static_target = STATIC_IMAGES_DIR / local_path.name
        if not static_target.exists():
            try:
                STATIC_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copyfile(local_path, static_target)
            except Exception:
                pass
        return f"/app/static/images/{local_path.name}"
    return url


def get_image_data_uri(url: Optional[str]) -> str:
    """Return inline Base64 data URI for instant, zero-network image rendering."""
    if not url:
        return ""
    if url in _DATA_URI_CACHE:
        return _DATA_URI_CACHE[url]

    local_path = get_local_image_path(url)
    if local_path and local_path.exists():
        try:
            with open(local_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                uri = f"data:image/jpeg;base64,{b64}"
                _DATA_URI_CACHE[url] = uri
                return uri
        except Exception:
            pass

    return url
