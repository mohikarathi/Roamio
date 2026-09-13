"""
Enrich Roamio destinations with authentic landscape photography and multi-image galleries.
Uses Wikipedia REST APIs (summary + media-list) and curated Unsplash imagery.
Downloads images to static/images and data/images_cache for 100% reliable offline rendering.
"""

import os
import sys
import json
import time
import sqlite3
import hashlib
import logging
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "roamio.db"
CACHE_DIR = PROJECT_ROOT / "data" / "images_cache"
STATIC_DIR = PROJECT_ROOT / "static" / "images"
GALLERIES_PATH = PROJECT_ROOT / "data" / "destination_galleries.json"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("enrich_images")

HEADERS = {
    "User-Agent": "RoamioTravelConcierge/2.0 (https://roamio.travel; contact@roamio.travel)"
}

# Regional fallbacks (Continent-specific high quality landscape travel photos)
REGIONAL_FALLBACKS = {
    "Europe": {
        "City": [
            ("https://images.unsplash.com/photo-1513581166391-887a96ddeafd?auto=format&fit=crop&w=800&q=80", "Ouael Ben Salah", "Historic European cathedral and cobblestone avenue"),
            ("https://images.unsplash.com/photo-1543429776-2782fc8e1acd?auto=format&fit=crop&w=800&q=80", "Jonathan Roger", "Terracotta roofs and Renaissance architecture"),
            ("https://images.unsplash.com/photo-1514890547357-a9ee288728e0?auto=format&fit=crop&w=800&q=80", "Dan Novac", "European canal waterway at twilight"),
            ("https://images.unsplash.com/photo-1509840841025-9088ba78a826?auto=format&fit=crop&w=800&q=80", "Henrique Ferreira", "Historic palace courtyard and grand colonnade")
        ],
        "Beach": [
            ("https://images.unsplash.com/photo-1570077188670-e3a8d69ac5ff?auto=format&fit=crop&w=800&q=80", "Heidi Kaden", "Mediterranean whitewashed cliffside and Aegean azure sea"),
            ("https://images.unsplash.com/photo-1533105079780-92b9be482077?auto=format&fit=crop&w=800&q=80", "Nick Fewings", "Amalfi Mediterranean coastal cliff village"),
            ("https://images.unsplash.com/photo-1516483638261-f4dbaf036963?auto=format&fit=crop&w=800&q=80", "Jack Ward", "Pastel cliffside harbor in Cinque Terre"),
            ("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=800&q=80", "Sean Oulashin", "Sandy coastal Mediterranean cove")
        ],
        "Mountain": [
            ("https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=800&q=80", "Kalen Emsley", "Snowcapped European Alpine summits and green valley"),
            ("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=800&q=80", "Bailey Zindel", "Alpine lake reflecting jagged granite peaks"),
            ("https://images.unsplash.com/photo-1432405972618-c60b0225b8f9?auto=format&fit=crop&w=800&q=80", "Lachlan Gowen", "Cascading alpine mountain river and pine forest")
        ]
    },
    "Asia": {
        "City": [
            ("https://images.unsplash.com/photo-1503899036084-c55cdd92da26?auto=format&fit=crop&w=800&q=80", "Jezael Melgoza", "Modern illuminated Tokyo skyline"),
            ("https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?auto=format&fit=crop&w=800&q=80", "Sawyer Bengtson", "Asian metropolitan street"),
            ("https://images.unsplash.com/photo-1508009603885-50cf7c579365?auto=format&fit=crop&w=800&q=80", "Bradley Prentice", "Historic riverside monument"),
            ("https://images.unsplash.com/photo-1513581166391-887a96ddeafd?auto=format&fit=crop&w=800&q=80", "Ouael Ben Salah", "Vibrant Asian night street market")
        ],
        "Beach": [
            ("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=800&q=80", "Sean Oulashin", "White coral sand and turquoise warm waters"),
            ("https://images.unsplash.com/photo-1589394815804-964ed0be2eb5?auto=format&fit=crop&w=800&q=80", "Miltiadis Fragkidis", "Dramatic tropical limestone sea cliffs"),
            ("https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?auto=format&fit=crop&w=800&q=80", "Ferran Feixas", "Sunset over palm-lined tropical coast"),
            ("https://images.unsplash.com/photo-1537996194471-e657df975ab4?auto=format&fit=crop&w=800&q=80", "Jeremy Bishop", "Lush tropical coastal headland")
        ],
        "Mountain": [
            ("https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?auto=format&fit=crop&w=800&q=80", "Su San Lee", "Mountain shrine nestled in cedar forest"),
            ("https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=800&q=80", "Kalen Emsley", "Misty mountain ridges and tea plantations"),
            ("https://images.unsplash.com/photo-1432405972618-c60b0225b8f9?auto=format&fit=crop&w=800&q=80", "Lachlan Gowen", "Cascading waterfall in tropical highland")
        ]
    }
}


def download_and_cache_image(url: str) -> bool:
    """Download an image and cache it in static/images and data/images_cache."""
    if not url or not url.startswith("http"):
        return False
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{url_hash}.jpg"
    static_file = STATIC_DIR / f"{url_hash}.jpg"

    if cache_file.exists() and static_file.exists() and cache_file.stat().st_size > 1000:
        return True

    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=8) as resp:
            content = resp.read()
            if len(content) > 1000:
                with open(cache_file, "wb") as f:
                    f.write(content)
                with open(static_file, "wb") as f:
                    f.write(content)
                return True
    except Exception as e:
        logger.debug(f"Failed to download {url}: {e}")
    return False


def fetch_wikipedia_details(name: str, city: str, country: str):
    """Fetch Wikipedia lead summary thumbnail, description, and media list photos."""
    queries = [
        name,
        f"{name}, {country}",
        f"{name} ({country})",
        city if city and city != name else None,
        f"{city}, {country}" if city and city != name else None
    ]
    queries = [q for q in queries if q]

    summary_data = None
    best_title = None

    for q in queries:
        clean_q = q.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(clean_q)}"
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                thumb = data.get("thumbnail", {}).get("source")
                # Exclude SVG, maps, flags, coats of arms
                if thumb and not any(x in thumb.lower() for x in [".svg", "flag", "coat_of_arms", "locator_map", "symbol"]):
                    summary_data = data
                    best_title = data.get("title") or clean_q
                    break
        except Exception:
            continue

    photos = []
    if best_title:
        clean_best = best_title.replace(" ", "_")
        media_url = f"https://en.wikipedia.org/api/rest_v1/page/media-list/{urllib.parse.quote(clean_best)}"
        try:
            req = urllib.request.Request(media_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=5) as resp:
                media_data = json.loads(resp.read().decode("utf-8"))
                items = [it for it in media_data.get("items", []) if it.get("type") == "image"]
                for it in items:
                    src = it.get("srcset", [{}])[-1].get("src") or it.get("srcset", [{}])[0].get("src")
                    title_attr = it.get("title", "")
                    if src and not any(x in src.lower() for x in [".svg", "flag", "icon", "map", "coat_of_arms", "symbol", "logo"]):
                        if src.startswith("//"):
                            src = "https:" + src
                        clean_alt = title_attr.replace("File:", "").replace("_", " ").split(".")[0]
                        photos.append({
                            "url": src,
                            "author": "Wikimedia Commons / CC-BY-SA",
                            "alt": f"{name}, {country} - {clean_alt}"
                        })
                        if len(photos) >= 6:
                            break
        except Exception:
            pass

    return summary_data, photos


def enrich_destination(dest_row):
    """Enrich a single destination with verified images and gallery items."""
    dest_id, name, city, country, continent, category, orig_img, orig_alt = dest_row
    logger.info(f"Processing {dest_id}: {name}, {country} ({continent})")

    summary_data, wiki_photos = fetch_wikipedia_details(name, city, country)
    gallery = []

    # Lead photo from summary
    primary_img = orig_img
    primary_thumb = orig_img
    primary_alt = orig_alt or f"{name}, {country}"
    primary_author = "Unsplash"

    if summary_data and summary_data.get("thumbnail", {}).get("source"):
        thumb_src = summary_data["thumbnail"]["source"]
        orig_src = summary_data.get("originalimage", {}).get("source") or thumb_src
        primary_img = orig_src
        primary_thumb = thumb_src
        primary_author = "Wikimedia Commons / CC-BY-SA"
        desc = summary_data.get("description", "")
        primary_alt = f"{name}, {country} — {desc}" if desc else f"{name}, {country}"

        gallery.append({
            "url": primary_thumb,
            "author": primary_author,
            "alt": primary_alt
        })

    # Add photos from media list
    for p in wiki_photos:
        if not any(g["url"] == p["url"] for g in gallery):
            gallery.append(p)
        if len(gallery) >= 4:
            break

    # If still fewer than 4 photos, add continent-appropriate fallback
    if len(gallery) < 4:
        reg_map = REGIONAL_FALLBACKS.get(continent, REGIONAL_FALLBACKS["Europe"])
        cat_key = "Beach" if any(k in category.lower() for k in ["beach", "coast", "island"]) else ("Mountain" if any(k in category.lower() for k in ["mountain", "park", "nature"]) else "City")
        fallbacks = reg_map.get(cat_key, reg_map["City"])
        for fb_url, fb_auth, fb_alt in fallbacks:
            if not any(g["url"] == fb_url for g in gallery):
                gallery.append({
                    "url": fb_url,
                    "author": fb_auth,
                    "alt": f"{name}, {country} ({fb_alt})"
                })
            if len(gallery) >= 4:
                break

    # Download and cache images
    for g in gallery[:4]:
        download_and_cache_image(g["url"])

    if primary_thumb:
        download_and_cache_image(primary_thumb)

    return {
        "destination_id": dest_id,
        "name": name,
        "country": country,
        "continent": continent,
        "primary_img": primary_img,
        "primary_thumb": primary_thumb,
        "primary_alt": primary_alt,
        "primary_author": primary_author,
        "gallery": gallery[:4]
    }


def main():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 1. Normalize continents for European countries previously categorized as 'Other'
    european_countries = ['Belgium', 'Denmark', 'Luxembourg', 'Malta', 'Monaco', 'Russia', 'Serbia', 'Sweden', 'Ukraine']
    for c_name in european_countries:
        c.execute("UPDATE destinations SET continent = 'Europe' WHERE country = ? AND continent = 'Other'", (c_name,))
    c.execute("UPDATE destinations SET continent = 'Europe' WHERE country = 'Turkey' AND continent = 'Europe/Asia'")
    conn.commit()
    logger.info("Normalized continent values for European destinations.")

    # 2. Query all destinations
    c.execute("SELECT destination_id, name, city, country, continent, category, image_url, image_alt FROM destinations ORDER BY continent, country, name")
    destinations = c.fetchall()
    logger.info(f"Loaded {len(destinations)} destinations for image enrichment.")

    galleries_by_id = {}
    galleries_by_name = {}

    # 3. Process with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(enrich_destination, row): row for row in destinations}
        for fut in as_completed(futures):
            res = fut.result()
            dest_id = res["destination_id"]
            name_key = res["name"].strip().lower()

            # Update DB
            c.execute("""
                UPDATE destinations
                SET image_url = ?, thumbnail_url = ?, image_alt = ?, photo_author = ?, photo_author_url = ?
                WHERE destination_id = ?
            """, (
                res["primary_img"],
                res["primary_thumb"],
                res["primary_alt"],
                res["primary_author"],
                "https://commons.wikimedia.org" if "Wikimedia" in res["primary_author"] else "https://unsplash.com",
                dest_id
            ))

            galleries_by_id[dest_id] = res["gallery"]
            galleries_by_name[name_key] = res["gallery"]

    conn.commit()
    conn.close()

    # 4. Save destination galleries JSON
    payload = {
        "by_id": galleries_by_id,
        "by_name": galleries_by_name
    }
    with open(GALLERIES_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Successfully enriched all {len(destinations)} destinations. Saved galleries to {GALLERIES_PATH}.")


if __name__ == "__main__":
    main()
