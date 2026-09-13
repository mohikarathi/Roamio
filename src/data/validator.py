"""Data validation module ensuring destination records meet strict quality standards."""

import logging
from typing import List, Tuple, Dict, Any
from src.data.models import Destination

logger = logging.getLogger(__name__)

# Valid ranges
VALID_LAT_RANGE = (-90.0, 90.0)
VALID_LON_RANGE = (-180.0, 180.0)
VALID_COST_LEVELS = {"Low", "Medium", "High", "Luxury"}
VALID_SAFETY_RATINGS = {"High", "Medium", "Low"}
MIN_DAILY_COST_INR = 500.0
MAX_DAILY_COST_INR = 200_000.0


class ValidationResult:
    """Stores validation pass/fail status and discovered issues."""
    def __init__(self, is_valid: bool, errors: List[str], warnings: List[str]):
        self.is_valid = is_valid
        self.errors = errors
        self.warnings = warnings

    def __repr__(self) -> str:
        return f"ValidationResult(is_valid={self.is_valid}, errors={len(self.errors)}, warnings={len(self.warnings)})"


def validate_destination(dest: Destination) -> ValidationResult:
    """
    Validate a canonical destination instance.
    Flags invalid coordinates, impossible costs, missing text, and malformed categories.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Identity & Name
    if not dest.destination_id or not dest.destination_id.strip():
        errors.append("Empty destination_id")
    if not dest.name or len(dest.name.strip()) < 2:
        errors.append(f"Destination name too short or missing: '{dest.name}'")
    if not dest.country or len(dest.country.strip()) < 2:
        errors.append(f"Invalid or missing country: '{dest.country}'")

    # 2. Coordinates
    if not (VALID_LAT_RANGE[0] <= dest.latitude <= VALID_LAT_RANGE[1]):
        errors.append(f"Latitude out of bounds [-90, 90]: {dest.latitude}")
    if not (VALID_LON_RANGE[0] <= dest.longitude <= VALID_LON_RANGE[1]):
        errors.append(f"Longitude out of bounds [-180, 180]: {dest.longitude}")
    if dest.latitude == 0.0 and dest.longitude == 0.0:
        warnings.append(f"Destination '{dest.name}' has (0, 0) coordinates (Null Island)")

    # 3. Description & Text Quality
    if not dest.description or len(dest.description.strip()) < 15:
        errors.append(f"Description too short or missing ({len(dest.description) if dest.description else 0} chars)")

    # 4. Financial & Costs
    if not (MIN_DAILY_COST_INR <= dest.est_daily_cost_inr <= MAX_DAILY_COST_INR):
        errors.append(f"Daily cost out of realistic bounds [₹{MIN_DAILY_COST_INR}, ₹{MAX_DAILY_COST_INR}]: ₹{dest.est_daily_cost_inr}")
    if dest.cost_level not in VALID_COST_LEVELS:
        warnings.append(f"Unrecognized cost level: '{dest.cost_level}'")

    # 5. Temporal / Seasonality
    if dest.best_months:
        invalid_months = [m for m in dest.best_months if not (1 <= m <= 12)]
        if invalid_months:
            errors.append(f"Invalid month numbers: {invalid_months}")

    # 6. Safety & Quality
    if dest.safety_rating not in VALID_SAFETY_RATINGS:
        warnings.append(f"Unusual safety rating: '{dest.safety_rating}'")

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


def validate_destination_batch(destinations: List[Destination]) -> Tuple[List[Destination], Dict[str, Any]]:
    """
    Validate an entire batch of destinations.
    Returns (valid_destinations, audit_report).
    """
    valid: List[Destination] = []
    rejected: List[Tuple[Destination, List[str]]] = []
    warnings_total: int = 0
    seen_ids = set()
    duplicate_ids: List[str] = []

    for d in destinations:
        # Check duplicate ID
        if d.destination_id in seen_ids:
            duplicate_ids.append(d.destination_id)
        seen_ids.add(d.destination_id)

        res = validate_destination(d)
        if res.is_valid:
            valid.append(d)
        else:
            rejected.append((d, res.errors))
        warnings_total += len(res.warnings)

    report = {
        "total_evaluated": len(destinations),
        "valid_count": len(valid),
        "rejected_count": len(rejected),
        "duplicate_ids": duplicate_ids,
        "warnings_count": warnings_total,
        "rejection_samples": [
            {"id": d.destination_id, "name": d.name, "errors": errs}
            for d, errs in rejected[:5]
        ]
    }

    if rejected:
        logger.warning(f"Batch validation: Rejected {len(rejected)} of {len(destinations)} records.")
    return valid, report
