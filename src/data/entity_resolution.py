"""Entity resolution for travel destinations across heterogeneous data sources."""

import math
from typing import List, Dict, Optional, Tuple
from src.data.models import Destination
from src.data.normalizer import clean_place_name, normalize_text


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance in km between two coordinate points."""
    r = 6371.0  # Earth's radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def jaccard_similarity(str1: str, str2: str) -> float:
    """Compute token-level Jaccard similarity between two strings."""
    tokens1 = set(normalize_text(str1).split())
    tokens2 = set(normalize_text(str2).split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)


class EntityResolver:
    """
    Resolves and links destinations from multiple sources to canonical entities.
    Handles variations like 'Kyoto', 'Kyoto City', 'Kyoto, Japan'.
    """

    def __init__(self, geo_threshold_km: float = 35.0, string_sim_threshold: float = 0.6):
        self.geo_threshold_km = geo_threshold_km
        self.string_sim_threshold = string_sim_threshold
        self.canonical_entities: Dict[str, Destination] = {}
        # Secondary index by (cleaned_name, country)
        self.name_country_index: Dict[Tuple[str, str], str] = {}
        # Secondary index by external_id
        self.external_id_index: Dict[str, str] = {}

    def is_match(self, dest1: Destination, dest2: Destination) -> Tuple[bool, str]:
        """
        Determine if two destination records refer to the same geographical entity.
        Returns (is_match, reason).
        """
        # Rule 1: Exact External ID match (e.g. Wikidata QID)
        if (dest1.external_id and dest2.external_id and 
            dest1.external_id == dest2.external_id):
            return True, f"Exact external ID match: {dest1.external_id}"

        # Rule 2: Country must match (or one is Unknown)
        country1_norm = normalize_text(dest1.country)
        country2_norm = normalize_text(dest2.country)
        same_country = (country1_norm == country2_norm or 
                        country1_norm in ["unknown", ""] or 
                        country2_norm in ["unknown", ""])

        # Rule 3: Normalized name + Country exact match
        clean_name1 = clean_place_name(dest1.name)
        clean_name2 = clean_place_name(dest2.name)

        if clean_name1 and clean_name1 == clean_name2 and same_country:
            return True, f"Normalized name & country match: {clean_name1} in {dest1.country}"

        # Rule 4: High string similarity + Same country + Small geo distance
        sim = jaccard_similarity(dest1.name, dest2.name)
        if same_country and sim >= self.string_sim_threshold:
            # Check coordinates if both are valid (non-zero)
            if (dest1.latitude != 0.0 or dest1.longitude != 0.0) and (dest2.latitude != 0.0 or dest2.longitude != 0.0):
                dist = haversine_distance_km(dest1.latitude, dest1.longitude, dest2.latitude, dest2.longitude)
                if dist <= self.geo_threshold_km:
                    return True, f"Geo proximity ({dist:.1f}km) and token similarity ({sim:.2f})"
            else:
                # Without coords, require very high token similarity
                if sim >= 0.85:
                    return True, f"High token similarity without coords ({sim:.2f})"

        return False, "No match"

    def merge_destinations(self, canonical: Destination, new_dest: Destination) -> Destination:
        """Merge a newly ingested destination into an existing canonical destination."""
        # Keep richer description
        if len(new_dest.description) > len(canonical.description):
            desc = new_dest.description
        else:
            desc = canonical.description

        # Combine cultural significance
        cult = canonical.cultural_significance
        if new_dest.cultural_significance and new_dest.cultural_significance not in (cult or ""):
            cult = f"{cult} {new_dest.cultural_significance}".strip() if cult else new_dest.cultural_significance

        # Union sets of tags, activities, famous foods
        tags = sorted(list(set(canonical.tags).union(set(new_dest.tags))))
        activities = sorted(list(set(canonical.activities).union(set(new_dest.activities))))
        famous_foods = sorted(list(set(canonical.famous_foods).union(set(new_dest.famous_foods))))
        best_months = sorted(list(set(canonical.best_months).union(set(new_dest.best_months))))
        best_seasons = sorted(list(set(canonical.best_seasons).union(set(new_dest.best_seasons))))

        # Preserve external ID if available
        ext_id = canonical.external_id or new_dest.external_id

        # Merge sources provenance
        sources = set(canonical.source.split(" + "))
        sources.add(new_dest.source)
        merged_source = " + ".join(sorted(list(sources)))

        # Update fields
        return canonical.model_copy(update={
            "description": desc,
            "cultural_significance": cult,
            "tags": tags,
            "activities": activities,
            "famous_foods": famous_foods,
            "best_months": best_months,
            "best_seasons": best_seasons,
            "external_id": ext_id,
            "source": merged_source,
            "popularity_score": max(canonical.popularity_score, new_dest.popularity_score),
            "annual_tourists": canonical.annual_tourists or new_dest.annual_tourists,
        })

    def resolve_and_add(self, new_dest: Destination) -> Tuple[Destination, bool]:
        """
        Attempt to resolve new destination against known entities.
        Returns (canonical_destination, was_merged).
        """
        # Check by external_id first
        if new_dest.external_id and new_dest.external_id in self.external_id_index:
            canon_id = self.external_id_index[new_dest.external_id]
            merged = self.merge_destinations(self.canonical_entities[canon_id], new_dest)
            self.canonical_entities[canon_id] = merged
            return merged, True

        # Check by (clean_name, country)
        clean_key = (clean_place_name(new_dest.name), normalize_text(new_dest.country))
        if clean_key in self.name_country_index:
            canon_id = self.name_country_index[clean_key]
            merged = self.merge_destinations(self.canonical_entities[canon_id], new_dest)
            self.canonical_entities[canon_id] = merged
            return merged, True

        # Check against all existing entities using fuzzy match
        for canon_id, existing_dest in self.canonical_entities.items():
            matched, _ = self.is_match(existing_dest, new_dest)
            if matched:
                merged = self.merge_destinations(existing_dest, new_dest)
                self.canonical_entities[canon_id] = merged
                if new_dest.external_id:
                    self.external_id_index[new_dest.external_id] = canon_id
                return merged, True

        # If no match, add as new canonical entity
        self.canonical_entities[new_dest.destination_id] = new_dest
        self.name_country_index[clean_key] = new_dest.destination_id
        if new_dest.external_id:
            self.external_id_index[new_dest.external_id] = new_dest.destination_id

        return new_dest, False
