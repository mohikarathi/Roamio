"""Unit tests for image resolution, caching, and database persistence."""

import pytest
from pathlib import Path
from src.data.models import Destination
from src.data.images.curated import resolve_curated_image
from src.data.images.pipeline import ImagePipeline
from src.data.db import get_destination_by_id, update_destination_image, upsert_destinations, init_db


def test_resolve_curated_image_exact_match():
    res = resolve_curated_image(
        name="Kyoto",
        city="Kyoto",
        country="Japan",
        category="Cultural",
        continent="Asia"
    )
    assert res is not None
    assert "image_url" in res
    assert "photo_author" in res
    assert res["image_provider"] == "Unsplash"
    assert "kyoto" in res["image_alt"].lower() or "torii" in res["image_alt"].lower()


def test_resolve_curated_image_category_fallback():
    res = resolve_curated_image(
        name="Mystic Shore",
        city=None,
        country="Nowhere",
        category="Beach",
        continent="Oceania"
    )
    assert res is not None
    assert "image_url" in res
    assert "unsplash.com" in res["image_url"]
    assert res["photo_author"] is not None


def test_resolve_curated_image_safe_default():
    res = resolve_curated_image(
        name="Unknown Spot",
        city=None,
        country="Unknownland",
        category="NonExistentCategoryXYZ",
        continent="Unknown"
    )
    assert res is not None
    assert "image_url" in res
    assert len(res["image_url"]) > 10


def test_destination_image_fields():
    dest = Destination(
        destination_id="TEST-IMG-01",
        name="Test City",
        country="Test Country",
        latitude=10.0,
        longitude=20.0,
        category="City",
        description="A test destination with scenic views.",
        image_url="https://images.unsplash.com/sample.jpg",
        photo_author="Jane Doe",
        image_provider="Unsplash"
    )
    assert dest.image_url == "https://images.unsplash.com/sample.jpg"
    assert dest.photo_author == "Jane Doe"
    assert dest.image_provider == "Unsplash"


def test_update_destination_image_isolated_db(tmp_path: Path):
    test_db = tmp_path / "test_roamio.db"
    init_db(test_db)

    temp_dest = Destination(
        destination_id="TEST-TEMP-001",
        name="Temp Paradise",
        country="Fictionland",
        latitude=12.0,
        longitude=34.0,
        category="Beach",
        description="Temporary test place for image update testing.",
        est_daily_cost_inr=5000.0
    )
    upsert_destinations([temp_dest], db_path=test_db)

    updated = update_destination_image(
        destination_id="TEST-TEMP-001",
        image_url="https://images.unsplash.com/test_paradise.jpg",
        thumbnail_url="https://images.unsplash.com/test_paradise_thumb.jpg",
        photo_author="Test Photographer",
        image_provider="Unsplash",
        db_path=test_db
    )
    assert updated is True

    d = get_destination_by_id("TEST-TEMP-001", db_path=test_db)
    assert d is not None
    assert d.image_url == "https://images.unsplash.com/test_paradise.jpg"
    assert d.photo_author == "Test Photographer"


def test_get_image_data_uri():
    from src.data.images.local_cache import get_image_data_uri
    from src.data.db import get_all_destinations
    dests = get_all_destinations()
    assert len(dests) > 0
    uri = get_image_data_uri(dests[0].thumbnail_url or dests[0].image_url)
    assert uri is not None
    assert uri.startswith("data:image/jpeg;base64,")
    assert len(uri) > 1000


def test_destination_gallery_multi_images():
    from src.data.images.gallery import get_destination_gallery
    from src.data.db import get_all_destinations
    dests = get_all_destinations()
    
    # Test Thailand destination gallery
    thai_dests = [d for d in dests if d.country == "Thailand"]
    assert len(thai_dests) >= 10
    
    for d in thai_dests[:3]:
        gallery = get_destination_gallery(d)
        assert len(gallery) >= 3
        for img in gallery:
            assert "data_uri" in img
            assert img["data_uri"].startswith("data:image/jpeg;base64,")
            assert "web_url" in img
            assert img["web_url"].startswith("/app/static/images/")
            assert "author" in img
            assert "alt" in img


def test_get_image_web_url():
    from src.data.images.local_cache import get_image_web_url
    from src.data.db import get_all_destinations
    dests = get_all_destinations()
    assert len(dests) > 0
    web_url = get_image_web_url(dests[0].thumbnail_url or dests[0].image_url)
    assert web_url.startswith("/app/static/images/")
    assert web_url.endswith(".jpg")


def test_destination_galleries_catalog_and_european_accuracy():
    from src.data.images.gallery import _load_destination_galleries, get_destination_gallery
    from src.data.db import get_all_destinations
    catalog = _load_destination_galleries()
    assert len(catalog.get("by_id", {})) >= 250

    dests = {d.name: d for d in get_all_destinations()}
    
    # Check Mont Saint-Michel
    msm = dests.get("Mont Saint-Michel")
    assert msm is not None
    gallery = get_destination_gallery(msm)
    assert len(gallery) >= 3
    # Verify no Thai/Asian keywords in Mont Saint-Michel alts
    for img in gallery:
        alt_lower = img["alt"].lower()
        assert "wat arun" not in alt_lower
        assert "bangkok" not in alt_lower
        assert "thailand" not in alt_lower

    # Check Vienna
    vienna = dests.get("Vienna")
    assert vienna is not None
    v_gallery = get_destination_gallery(vienna)
    assert len(v_gallery) >= 3
    for img in v_gallery:
        alt_lower = img["alt"].lower()
        assert "wat arun" not in alt_lower
        assert "bangkok" not in alt_lower
        assert "thailand" not in alt_lower



