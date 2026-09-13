"""Multi-source ingestion pipeline converting raw seed data and curated datasets into canonical models."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from src.config import SEED_EXCEL_PATH, RAW_DATA_DIR
from src.data.models import Destination
from src.data.curated_data import CURATED_GLOBAL_DESTINATIONS
from src.data.normalizer import (
    parse_tourist_count,
    parse_best_time,
    map_cost_to_inr,
    infer_continent,
    normalize_text
)

logger = logging.getLogger(__name__)


def ingest_seed_excel(excel_path: Optional[Path] = None) -> List[Destination]:
    """
    Ingest the baseline Excel dataset (209 rows) and normalize into Destination models.
    Fills missing values and preserves provenance.
    """
    path = excel_path or SEED_EXCEL_PATH
    if not path.exists():
        logger.warning(f"Seed file not found at {path}")
        return []

    df = pd.read_excel(path)
    # Standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    destinations: List[Destination] = []
    raw_records: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        name = str(row.get("destination", "Unknown")).strip()
        country = str(row.get("country", "Unknown")).strip()
        region = str(row.get("region", "")).strip() or None
        category = str(row.get("category", "City")).strip()

        # Coordinates
        lat = float(row.get("latitude", 0.0))
        lon = float(row.get("longitude", 0.0))

        # Tourist numbers
        tourists = parse_tourist_count(row.get("approximate_annual_tourists"))

        # Best time
        best_time_str = str(row.get("best_time_to_visit", "Any time"))
        months, seasons = parse_best_time(best_time_str)

        # Cost
        cost_str = str(row.get("cost_of_living", "Medium"))
        cost_level, daily_cost_inr = map_cost_to_inr(cost_str, country)

        # Safety
        safety_str = str(row.get("safety", "Generally safe"))
        if any(w in safety_str.lower() for w in ["bear", "conflict", "risk", "restricted", "low"]):
            safety_rating = "Low"
        else:
            safety_rating = "High"

        # Foods
        foods_raw = str(row.get("famous_foods", ""))
        foods = [f.strip() for f in foods_raw.split(",") if f.strip() and f.strip().lower() != "nan"]

        # Cultural description & narrative
        cult_sig = str(row.get("cultural_significance", "")).strip()
        if cult_sig.lower() == "nan":
            cult_sig = None

        desc = str(row.get("description", "")).strip()
        if not desc or desc.lower() == "nan":
            # Fallback narrative based on available features
            desc = f"{name} is a renowned {category.lower()} in {country}, known for its {cult_sig or 'rich heritage and scenic attractions'}."

        # Tags
        tags = [category.lower()]
        religion = str(row.get("majority_religion", "")).strip()
        if religion and religion.lower() != "nan":
            tags.append(religion.lower())
        language = str(row.get("language", "")).strip()
        if language and language.lower() != "nan":
            tags.append(f"{language.lower()} speaking")

        continent = infer_continent(country)
        dest_id = f"SEED-{idx + 1:04d}"

        dest = Destination(
            destination_id=dest_id,
            name=name,
            city=name if category.lower() in ["city", "coastal city", "town"] else None,
            region=region,
            country=country,
            continent=continent,
            latitude=lat,
            longitude=lon,
            category=category,
            tags=tags,
            description=desc,
            cultural_significance=cult_sig,
            activities=[f"explore {name}", "sightseeing", "cultural touring"],
            famous_foods=foods,
            cost_level=cost_level,
            est_daily_cost_inr=daily_cost_inr,
            currency=str(row.get("currency", "Local")),
            safety_rating=safety_rating,
            best_months=months,
            best_seasons=seasons,
            annual_tourists=tourists,
            popularity_score=min(1.0, (tourists or 1_000_000) / 20_000_000),
            source="Seed: destinations.xlsx"
        )
        destinations.append(dest)
        raw_records.append(row.to_dict())

    # Save raw seed artifact
    raw_seed_path = RAW_DATA_DIR / "seed_destinations.json"
    with open(raw_seed_path, "w", encoding="utf-8") as f:
        json.dump(raw_records, f, indent=2, default=str)

    logger.info(f"Ingested {len(destinations)} records from seed Excel file.")
    return destinations


def ingest_curated_dataset() -> List[Destination]:
    """Ingest multi-source curated global destinations dataset."""
    destinations: List[Destination] = []

    for idx, d in enumerate(CURATED_GLOBAL_DESTINATIONS):
        dest_id = f"CURATED-{idx + 1:04d}"
        dest = Destination(
            destination_id=dest_id,
            name=d["name"],
            city=d.get("city"),
            region=d.get("region"),
            country=d["country"],
            continent=d.get("continent", infer_continent(d["country"])),
            latitude=d["latitude"],
            longitude=d["longitude"],
            category=d["category"],
            tags=d.get("tags", []),
            description=d["description"],
            cultural_significance=d.get("cultural_significance"),
            activities=d.get("activities", []),
            famous_foods=d.get("famous_foods", []),
            cost_level=d.get("cost_level", "Medium"),
            est_daily_cost_inr=d.get("est_daily_cost_inr", 5000.0),
            currency=d.get("currency", "USD"),
            safety_rating=d.get("safety_rating", "High"),
            best_months=d.get("best_months", list(range(1, 13))),
            best_seasons=d.get("best_seasons", ["Year-round"]),
            annual_tourists=d.get("annual_tourists"),
            popularity_score=d.get("popularity_score", 0.9),
            source=d.get("source", "Curated OpenTravel"),
            external_id=d.get("external_id")
        )
        destinations.append(dest)

    # Save raw curated artifact
    raw_curated_path = RAW_DATA_DIR / "curated_destinations.json"
    with open(raw_curated_path, "w", encoding="utf-8") as f:
        json.dump(CURATED_GLOBAL_DESTINATIONS, f, indent=2, default=str)

    logger.info(f"Ingested {len(destinations)} records from curated global sources.")
    return destinations
