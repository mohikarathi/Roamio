"""Automated image pipeline for Roamio destinations.

Fetches high-resolution landscape travel imagery for destinations using Unsplash/Pexels API
when keys are available, and reliably falls back to curated verified travel photography.
Ensures 100% catalog coverage with zero broken image placeholders.
"""

import os
import json
import logging
import argparse
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.config import DATABASE_PATH
from src.data.db import get_all_destinations, update_destination_image, init_db
from src.data.images.curated import resolve_curated_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_PATH = Path("data/images.json")


class ImagePipeline:
    """Image acquisition, normalization, caching, and persistence pipeline."""

    def __init__(self, provider: str = "auto"):
        self.provider = provider
        self.unsplash_key = os.environ.get("UNSPLASH_ACCESS_KEY")
        self.pexels_key = os.environ.get("PEXELS_API_KEY")
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if CACHE_PATH.exists():
            try:
                with open(CACHE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not parse existing cache at {CACHE_PATH}: {e}")
        return {}

    def _save_cache(self) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved image metadata cache ({len(self.cache)} entries) to {CACHE_PATH}")

    def fetch_unsplash(self, query: str) -> Optional[Dict[str, str]]:
        """Query Unsplash search API for landscape photo."""
        if not self.unsplash_key:
            return None
        try:
            url = f"https://api.unsplash.com/search/photos?query={urllib.parse.quote(query)}&orientation=landscape&per_page=1"
            req = urllib.request.Request(url, headers={"Authorization": f"Client-ID {self.unsplash_key}"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = data.get("results", [])
                if results:
                    p = results[0]
                    user = p.get("user", {})
                    return {
                        "image_url": p.get("urls", {}).get("regular"),
                        "thumbnail_url": p.get("urls", {}).get("small"),
                        "photo_author": user.get("name", "Unsplash Contributor"),
                        "photo_author_url": user.get("links", {}).get("html", "https://unsplash.com"),
                        "image_provider": "Unsplash",
                        "image_alt": p.get("alt_description") or query
                    }
        except Exception as e:
            logger.warning(f"Unsplash API request failed for query '{query}': {e}")
        return None

    def fetch_pexels(self, query: str) -> Optional[Dict[str, str]]:
        """Query Pexels search API for landscape photo."""
        if not self.pexels_key:
            return None
        try:
            url = f"https://api.pexels.com/v1/search?query={urllib.parse.quote(query)}&orientation=landscape&per_page=1"
            req = urllib.request.Request(url, headers={"Authorization": self.pexels_key})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                photos = data.get("photos", [])
                if photos:
                    p = photos[0]
                    return {
                        "image_url": p.get("src", {}).get("large2x") or p.get("src", {}).get("landscape"),
                        "thumbnail_url": p.get("src", {}).get("medium"),
                        "photo_author": p.get("photographer", "Pexels Photographer"),
                        "photo_author_url": p.get("photographer_url", "https://pexels.com"),
                        "image_provider": "Pexels",
                        "image_alt": p.get("alt") or query
                    }
        except Exception as e:
            logger.warning(f"Pexels API request failed for query '{query}': {e}")
        return None

    def get_image_for_destination(self, dest) -> Dict[str, str]:
        """Retrieve landscape photo metadata with multiple fallback tiers."""
        # 1. Check local cache
        if dest.destination_id in self.cache:
            entry = self.cache[dest.destination_id]
            if entry.get("image_url"):
                return entry

        # 2. Try external APIs if keys are available and not disabled
        query = f"{dest.name} {dest.country} landscape travel"
        if self.provider in ("auto", "unsplash") and self.unsplash_key:
            res = self.fetch_unsplash(query)
            if res:
                return res

        if self.provider in ("auto", "pexels") and self.pexels_key:
            res = self.fetch_pexels(query)
            if res:
                return res

        # 3. Use curated high-resolution registry
        curated = resolve_curated_image(
            name=dest.name,
            city=dest.city,
            country=dest.country,
            category=dest.category,
            continent=dest.continent
        )
        return curated

    def run(
        self,
        missing_only: bool = False,
        destination_id: Optional[str] = None,
        force: bool = False,
        dry_run: bool = False
    ) -> Dict[str, int]:
        """Execute pipeline over catalog destinations."""
        init_db()
        dests = get_all_destinations()
        logger.info(f"Starting image pipeline for {len(dests)} destinations in database...")

        if destination_id:
            dests = [d for d in dests if d.destination_id == destination_id]
            if not dests:
                logger.error(f"No destination found with ID {destination_id}")
                return {"processed": 0, "updated": 0}

        stats = {"processed": 0, "updated": 0, "skipped": 0}

        for d in dests:
            stats["processed"] += 1
            has_image = bool(d.image_url)

            if missing_only and has_image and not force:
                stats["skipped"] += 1
                continue

            metadata = self.get_image_for_destination(d)
            self.cache[d.destination_id] = metadata

            if not dry_run:
                update_destination_image(
                    destination_id=d.destination_id,
                    image_url=metadata["image_url"],
                    thumbnail_url=metadata.get("thumbnail_url", metadata["image_url"]),
                    photo_author=metadata.get("photo_author"),
                    photo_author_url=metadata.get("photo_author_url"),
                    image_provider=metadata.get("image_provider", "Unsplash"),
                    image_alt=metadata.get("image_alt", f"{d.name}, {d.country}")
                )
                stats["updated"] += 1

        if not dry_run:
            self._save_cache()

        logger.info(f"Image pipeline completed: {stats}")
        return stats


def fetch_and_populate_images(missing_only: bool = True) -> Dict[str, int]:
    """Programmatic interface to run the image pipeline."""
    pipeline = ImagePipeline(provider="auto")
    return pipeline.run(missing_only=missing_only)


def main():
    parser = argparse.ArgumentParser(description="Roamio automated destination imagery pipeline")
    parser.add_argument("--missing-only", action="store_true", help="Only populate destinations missing image URLs")
    parser.add_argument("--destination-id", type=str, default=None, help="Process single destination by canonical ID")
    parser.add_argument("--force", action="store_true", help="Force re-fetch/overwrite existing records")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without saving to database")
    parser.add_argument("--provider", type=str, default="auto", choices=["auto", "unsplash", "pexels", "curated"], help="Image search provider")
    args = parser.parse_args()

    pipeline = ImagePipeline(provider=args.provider)
    pipeline.run(
        missing_only=args.missing_only,
        destination_id=args.destination_id,
        force=args.force,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
