"""SQLite database layer for Roamio destinations and provenance tracking."""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from src.config import DATABASE_PATH
from src.data.models import Destination

logger = logging.getLogger(__name__)


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return a SQLite connection with row factory enabled."""
    path = db_path or DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize SQLite schema if it does not already exist."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Destinations table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS destinations (
        destination_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        city TEXT,
        region TEXT,
        country TEXT NOT NULL,
        continent TEXT NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        category TEXT NOT NULL,
        tags_json TEXT,
        description TEXT NOT NULL,
        cultural_significance TEXT,
        activities_json TEXT,
        famous_foods_json TEXT,
        cost_level TEXT NOT NULL,
        est_daily_cost_inr REAL NOT NULL,
        currency TEXT,
        safety_rating TEXT,
        best_months_json TEXT,
        best_seasons_json TEXT,
        annual_tourists INTEGER,
        popularity_score REAL,
        source TEXT,
        external_id TEXT,
        image_url TEXT,
        thumbnail_url TEXT,
        photo_author TEXT,
        photo_author_url TEXT,
        image_provider TEXT,
        image_alt TEXT,
        last_updated TEXT
    )
    """)

    # Ensure schema migration for existing databases
    cursor.execute("PRAGMA table_info(destinations)")
    existing_cols = {row["name"] for row in cursor.fetchall()}
    for col, col_type in [
        ("image_url", "TEXT"),
        ("thumbnail_url", "TEXT"),
        ("photo_author", "TEXT"),
        ("photo_author_url", "TEXT"),
        ("image_provider", "TEXT"),
        ("image_alt", "TEXT"),
    ]:
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE destinations ADD COLUMN {col} {col_type}")

    # Indices for high-frequency filtering
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_country ON destinations(country)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_continent ON destinations(continent)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON destinations(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cost ON destinations(est_daily_cost_inr)")

    # Data sources provenance table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sources (
        source_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        license TEXT,
        update_frequency TEXT,
        record_count INTEGER,
        last_updated TEXT
    )
    """)

    conn.commit()
    conn.close()
    logger.info("Database schema initialized successfully.")


def _row_to_destination(row: sqlite3.Row) -> Destination:
    """Convert SQLite row to canonical Destination instance."""
    col_keys = row.keys()
    return Destination(
        destination_id=row["destination_id"],
        name=row["name"],
        city=row["city"],
        region=row["region"],
        country=row["country"],
        continent=row["continent"],
        latitude=row["latitude"],
        longitude=row["longitude"],
        category=row["category"],
        tags=json.loads(row["tags_json"] or "[]"),
        description=row["description"],
        cultural_significance=row["cultural_significance"],
        activities=json.loads(row["activities_json"] or "[]"),
        famous_foods=json.loads(row["famous_foods_json"] or "[]"),
        cost_level=row["cost_level"],
        est_daily_cost_inr=row["est_daily_cost_inr"],
        currency=row["currency"],
        safety_rating=row["safety_rating"],
        best_months=json.loads(row["best_months_json"] or "[]"),
        best_seasons=json.loads(row["best_seasons_json"] or "[]"),
        annual_tourists=row["annual_tourists"],
        popularity_score=row["popularity_score"],
        source=row["source"],
        external_id=row["external_id"],
        image_url=row["image_url"] if "image_url" in col_keys else None,
        thumbnail_url=row["thumbnail_url"] if "thumbnail_url" in col_keys else None,
        photo_author=row["photo_author"] if "photo_author" in col_keys else None,
        photo_author_url=row["photo_author_url"] if "photo_author_url" in col_keys else None,
        image_provider=row["image_provider"] if "image_provider" in col_keys else "Unsplash",
        image_alt=row["image_alt"] if "image_alt" in col_keys else None,
        last_updated=row["last_updated"]
    )


def upsert_destinations(destinations: List[Destination], db_path: Optional[Path] = None) -> int:
    """Upsert a list of Destination records into the SQLite database."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    query = """
    INSERT INTO destinations (
        destination_id, name, city, region, country, continent,
        latitude, longitude, category, tags_json, description,
        cultural_significance, activities_json, famous_foods_json,
        cost_level, est_daily_cost_inr, currency, safety_rating,
        best_months_json, best_seasons_json, annual_tourists,
        popularity_score, source, external_id,
        image_url, thumbnail_url, photo_author, photo_author_url, image_provider, image_alt,
        last_updated
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(destination_id) DO UPDATE SET
        name=excluded.name,
        city=excluded.city,
        region=excluded.region,
        country=excluded.country,
        continent=excluded.continent,
        latitude=excluded.latitude,
        longitude=excluded.longitude,
        category=excluded.category,
        tags_json=excluded.tags_json,
        description=excluded.description,
        cultural_significance=excluded.cultural_significance,
        activities_json=excluded.activities_json,
        famous_foods_json=excluded.famous_foods_json,
        cost_level=excluded.cost_level,
        est_daily_cost_inr=excluded.est_daily_cost_inr,
        currency=excluded.currency,
        safety_rating=excluded.safety_rating,
        best_months_json=excluded.best_months_json,
        best_seasons_json=excluded.best_seasons_json,
        annual_tourists=excluded.annual_tourists,
        popularity_score=excluded.popularity_score,
        source=excluded.source,
        external_id=excluded.external_id,
        image_url=excluded.image_url,
        thumbnail_url=excluded.thumbnail_url,
        photo_author=excluded.photo_author,
        photo_author_url=excluded.photo_author_url,
        image_provider=excluded.image_provider,
        image_alt=excluded.image_alt,
        last_updated=excluded.last_updated
    """

    params = []
    for d in destinations:
        params.append((
            d.destination_id, d.name, d.city, d.region, d.country, d.continent,
            d.latitude, d.longitude, d.category, json.dumps(d.tags), d.description,
            d.cultural_significance, json.dumps(d.activities), json.dumps(d.famous_foods),
            d.cost_level, d.est_daily_cost_inr, d.currency, d.safety_rating,
            json.dumps(d.best_months), json.dumps(d.best_seasons), d.annual_tourists,
            d.popularity_score, d.source, d.external_id,
            d.image_url, d.thumbnail_url, d.photo_author, d.photo_author_url, d.image_provider, d.image_alt,
            d.last_updated
        ))

    cursor.executemany(query, params)
    conn.commit()
    count = cursor.rowcount
    conn.close()
    return count


def update_destination_image(
    destination_id: str,
    image_url: str,
    thumbnail_url: Optional[str] = None,
    photo_author: Optional[str] = None,
    photo_author_url: Optional[str] = None,
    image_provider: str = "Unsplash",
    image_alt: Optional[str] = None,
    db_path: Optional[Path] = None
) -> bool:
    """Update image metadata for a single destination."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE destinations SET
        image_url = ?,
        thumbnail_url = ?,
        photo_author = ?,
        photo_author_url = ?,
        image_provider = ?,
        image_alt = ?
    WHERE destination_id = ?
    """, (
        image_url,
        thumbnail_url or image_url,
        photo_author,
        photo_author_url,
        image_provider,
        image_alt,
        destination_id
    ))
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()
    return updated


def get_all_destinations(db_path: Optional[Path] = None) -> List[Destination]:
    """Retrieve all destinations from the database."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM destinations ORDER BY popularity_score DESC, name ASC")
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_destination(r) for r in rows]


def get_destination_by_id(destination_id: str, db_path: Optional[Path] = None) -> Optional[Destination]:
    """Retrieve a single destination by ID."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM destinations WHERE destination_id = ?", (destination_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return _row_to_destination(row)
    return None


def filter_destinations_query(
    continent: Optional[Union[str, List[str]]] = None,
    country: Optional[Union[str, List[str]]] = None,
    category: Optional[str] = None,
    max_daily_cost_inr: Optional[float] = None,
    travel_month: Optional[int] = None,
    min_safety: Optional[str] = None,
    db_path: Optional[Path] = None
) -> List[Destination]:
    """Execute SQL filtering with structured constraints."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    clauses = ["1=1"]
    params: List[Any] = []

    if continent and continent != "All":
        if isinstance(continent, (list, tuple, set)):
            cont_list = [c for c in continent if c and c != "All"]
            if cont_list:
                placeholders = ", ".join(["?"] * len(cont_list))
                clauses.append(f"continent IN ({placeholders})")
                params.extend(cont_list)
        else:
            clauses.append("continent = ?")
            params.append(continent)

    if country and country != "All":
        if isinstance(country, (list, tuple, set)):
            country_list = [c for c in country if c and c != "All"]
            if country_list:
                placeholders = ", ".join(["?"] * len(country_list))
                clauses.append(f"country IN ({placeholders})")
                params.extend(country_list)
        else:
            clauses.append("country = ?")
            params.append(country)

    if category and category != "All":
        clauses.append("category = ?")
        params.append(category)

    if max_daily_cost_inr is not None and max_daily_cost_inr > 0:
        clauses.append("est_daily_cost_inr <= ?")
        params.append(max_daily_cost_inr)

    if min_safety and min_safety != "All":
        if min_safety == "High":
            clauses.append("safety_rating = 'High'")
        elif min_safety == "Medium":
            clauses.append("safety_rating IN ('High', 'Medium')")

    query = f"SELECT * FROM destinations WHERE {' AND '.join(clauses)}"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    destinations = [_row_to_destination(r) for r in rows]

    # Post-filter on travel_month if provided (JSON column)
    if travel_month and 1 <= travel_month <= 12:
        destinations = [d for d in destinations if not d.best_months or travel_month in d.best_months]

    return destinations


def get_database_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Return database health metrics and catalog distribution."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM destinations")
    total_destinations = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT country) FROM destinations")
    total_countries = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT continent) FROM destinations")
    total_continents = cursor.fetchone()[0]

    cursor.execute("SELECT category, COUNT(*) as cnt FROM destinations GROUP BY category ORDER BY cnt DESC")
    category_counts = {r[0]: r[1] for r in cursor.fetchall()}

    cursor.execute("SELECT continent, COUNT(*) as cnt FROM destinations GROUP BY continent ORDER BY cnt DESC")
    continent_counts = {r[0]: r[1] for r in cursor.fetchall()}

    cursor.execute("SELECT cost_level, COUNT(*) as cnt FROM destinations GROUP BY cost_level ORDER BY cnt DESC")
    cost_counts = {r[0]: r[1] for r in cursor.fetchall()}

    conn.close()
    return {
        "total_destinations": total_destinations,
        "total_countries": total_countries,
        "total_continents": total_continents,
        "category_distribution": category_counts,
        "continent_distribution": continent_counts,
        "cost_distribution": cost_counts
    }
