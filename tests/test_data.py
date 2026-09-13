"""Unit tests for data normalization, validation, and entity resolution."""

import pytest
from src.data.models import Destination
from src.data.normalizer import (
    normalize_text,
    clean_place_name,
    parse_tourist_count,
    parse_best_time,
    map_cost_to_inr
)
from src.data.validator import validate_destination
from src.data.entity_resolution import EntityResolver, haversine_distance_km


def test_normalize_text():
    assert normalize_text("Café de Paris") == "cafe de paris"
    assert normalize_text("  KYOTO,   JAPAN!  ") == "kyoto, japan!"
    assert normalize_text("") == ""


def test_clean_place_name():
    assert clean_place_name("Kyoto City") == "kyoto"
    assert clean_place_name("Bali Island") == "bali"
    assert clean_place_name("Quebec Province") == "quebec"


def test_parse_tourist_count():
    assert parse_tourist_count("14 million") == 14_000_000
    assert parse_tourist_count("1.5 million") == 1_500_000
    assert parse_tourist_count("800,000") == 800_000
    assert parse_tourist_count("35-40 million") == 37_500_000
    assert parse_tourist_count(5000000) == 5_000_000
    assert parse_tourist_count(None) is None


def test_parse_best_time():
    months, seasons = parse_best_time("Spring (April-May) or Fall (Sept-Oct)")
    assert 4 in months and 5 in months and 9 in months and 10 in months
    assert "Spring" in seasons and "Fall" in seasons

    months_winter, seasons_winter = parse_best_time("Winter (Dec-Mar)")
    assert 12 in months_winter and 1 in months_winter and 2 in months_winter
    assert "Winter" in seasons_winter


def test_map_cost_to_inr():
    level_low, cost_low = map_cost_to_inr("Low")
    assert level_low == "Low"
    assert cost_low == 3000.0

    level_med, cost_med = map_cost_to_inr("Medium-high")
    assert level_med == "Medium"
    assert cost_med == 6500.0

    level_lux, cost_lux = map_cost_to_inr("Extremely High")
    assert level_lux == "Luxury"
    assert cost_lux == 18000.0


def test_destination_validation():
    # Valid destination
    valid_dest = Destination(
        destination_id="TEST-001",
        name="Kyoto",
        country="Japan",
        continent="Asia",
        latitude=35.0116,
        longitude=135.7681,
        category="Cultural",
        description="Historical imperial capital with ancient shrines, zen gardens, and temples.",
        cost_level="Medium",
        est_daily_cost_inr=5000.0
    )
    res = validate_destination(valid_dest)
    assert res.is_valid is True
    assert len(res.errors) == 0

    # Invalid destination: coordinates out of bounds, short description, negative cost
    invalid_dest = Destination(
        destination_id="TEST-002",
        name="X",
        country="Japan",
        continent="Asia",
        latitude=120.0,  # invalid latitude > 90
        longitude=200.0, # invalid longitude > 180
        category="City",
        description="Short",
        cost_level="Medium",
        est_daily_cost_inr=50.0  # below minimum
    )
    res_inv = validate_destination(invalid_dest)
    assert res_inv.is_valid is False
    assert len(res_inv.errors) >= 3


def test_entity_resolution():
    resolver = EntityResolver(geo_threshold_km=35.0, string_sim_threshold=0.6)

    d1 = Destination(
        destination_id="DEST-01",
        name="Kyoto",
        country="Japan",
        continent="Asia",
        latitude=35.0116,
        longitude=135.7681,
        category="Cultural",
        description="Ancient capital of Japan with Buddhist shrines.",
        external_id="Q34647",
        source="SourceA"
    )

    d2 = Destination(
        destination_id="DEST-02",
        name="Kyoto City",
        country="Japan",
        continent="Asia",
        latitude=35.0120,
        longitude=135.7690,
        category="City",
        description="Famous cultural center with over a thousand temples.",
        external_id="Q34647",
        source="SourceB"
    )

    _, was_merged1 = resolver.resolve_and_add(d1)
    assert was_merged1 is False

    merged, was_merged2 = resolver.resolve_and_add(d2)
    assert was_merged2 is True
    assert merged.destination_id == "DEST-01"
    assert "SourceA" in merged.source and "SourceB" in merged.source
