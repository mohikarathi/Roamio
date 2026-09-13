"""Text, geographic, and temporal normalization functions."""

import math
import re
import unicodedata
from typing import List, Tuple, Optional, Any

# Country to continent mapping
COUNTRY_TO_CONTINENT = {
    "Italy": "Europe", "France": "Europe", "Spain": "Europe", "Greece": "Europe",
    "United Kingdom": "Europe", "Germany": "Europe", "Switzerland": "Europe",
    "Portugal": "Europe", "Netherlands": "Europe", "Austria": "Europe",
    "Norway": "Europe", "Iceland": "Europe", "Croatia": "Europe",
    "Japan": "Asia", "Thailand": "Asia", "India": "Asia", "Indonesia": "Asia",
    "Vietnam": "Asia", "China": "Asia", "South Korea": "Asia", "Singapore": "Asia",
    "Malaysia": "Asia", "Sri Lanka": "Asia", "Nepal": "Asia", "Maldives": "Asia",
    "United Arab Emirates": "Middle East", "Turkey": "Europe/Asia", "Jordan": "Middle East",
    "United States": "North America", "Canada": "North America", "Mexico": "North America",
    "Costa Rica": "North America", "Peru": "South America", "Brazil": "South America",
    "Argentina": "South America", "Chile": "South America", "Colombia": "South America",
    "Egypt": "Africa", "Morocco": "Africa", "South Africa": "Africa", "Kenya": "Africa",
    "Tanzania": "Africa", "Australia": "Oceania", "New Zealand": "Oceania", "Fiji": "Oceania"
}

# Month mapping
MONTH_NAMES = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12
}

SEASON_MONTHS = {
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "fall": [9, 10, 11],
    "autumn": [9, 10, 11],
    "winter": [12, 1, 2]
}


def normalize_text(text: Optional[str]) -> str:
    """Normalize text by stripping diacritics, lowercase, removing special punctuation."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = text.lower().strip()
    # remove excessive spaces
    text = re.sub(r"\s+", " ", text)
    return text


def clean_place_name(name: str) -> str:
    """Clean a location name for entity resolution (e.g. 'Kyoto City' -> 'kyoto')."""
    norm = normalize_text(name)
    # Strip common administrative noise words
    noise_patterns = [r"\bcity\b", r"\btown\b", r"\bprovince\b", r"\bregion\b", r"\bprefecture\b", r"\bisland\b"]
    for pat in noise_patterns:
        norm = re.sub(pat, "", norm)
    return re.sub(r"\s+", " ", norm).strip()


def parse_tourist_count(val: Any) -> Optional[int]:
    """Parse annual tourist count from various string or numeric formats."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return int(val) if not math.isnan(val) else None
    
    s = str(val).lower().replace(",", "").strip()
    
    # Handle patterns like "14 million", "1.5 million", "35-40 million"
    range_match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*million", s)
    if range_match:
        avg = (float(range_match.group(1)) + float(range_match.group(2))) / 2.0
        return int(avg * 1_000_000)
        
    million_match = re.search(r"(\d+(?:\.\d+)?)\s*million", s)
    if million_match:
        return int(float(million_match.group(1)) * 1_000_000)
        
    k_match = re.search(r"(\d+(?:\.\d+)?)\s*k", s)
    if k_match:
        return int(float(k_match.group(1)) * 1_000)
        
    # Digits only
    digits = re.search(r"(\d+)", s)
    if digits:
        return int(digits.group(1))
        
    return None


def parse_best_time(text: Optional[str]) -> Tuple[List[int], List[str]]:
    """
    Parse natural language best time string into list of month integers (1-12)
    and list of seasons ('Spring', 'Summer', 'Fall', 'Winter').
    """
    if not text:
        return list(range(1, 13)), ["Year-round"]
        
    s = text.lower()
    months = set()
    seasons = set()
    
    if "year-round" in s or "any time" in s:
        return list(range(1, 13)), ["Year-round"]
        
    # Look for season mentions
    for season, m_list in SEASON_MONTHS.items():
        if season in s:
            seasons.add(season.capitalize())
            months.update(m_list)
            
    # Look for month ranges or individual months: e.g. "April-May", "Sept-Oct"
    month_range_pattern = r"([a-z]+)\s*[-–]\s*([a-z]+)"
    for m in re.finditer(month_range_pattern, s):
        start_m, end_m = m.group(1), m.group(2)
        if start_m in MONTH_NAMES and end_m in MONTH_NAMES:
            start_idx = MONTH_NAMES[start_m]
            end_idx = MONTH_NAMES[end_m]
            if start_idx <= end_idx:
                months.update(range(start_idx, end_idx + 1))
            else:  # Wraps around year (e.g. Dec-Mar)
                months.update(range(start_idx, 13))
                months.update(range(1, end_idx + 1))
                
    # Also individual months mentioned
    for word in re.findall(r"\b[a-z]+\b", s):
        if word in MONTH_NAMES:
            months.add(MONTH_NAMES[word])
            
    if not months:
        months = set(range(1, 13))
    if not seasons:
        seasons = {"Year-round"}
        
    return sorted(list(months)), sorted(list(seasons))


def map_cost_to_inr(cost_str: Optional[str], country: str = "") -> Tuple[str, float]:
    """
    Map cost category to normalized daily cost in INR.
    Low: ₹2,000 - ₹3,500
    Medium: ₹4,000 - ₹7,000
    High: ₹8,000 - ₹14,000
    Extremely High / Luxury: ₹15,000 - ₹25,000
    """
    if not cost_str:
        return "Medium", 5000.0
        
    s = cost_str.lower().strip()
    if "free" in s or "very low" in s:
        return "Low", 2000.0
    elif "low" in s:
        return "Low", 3000.0
    elif "medium-high" in s:
        return "Medium", 6500.0
    elif "medium" in s or "varies" in s:
        return "Medium", 5000.0
    elif "extremely high" in s or "luxury" in s:
        return "Luxury", 18000.0
    elif "high" in s:
        return "High", 10000.0
    else:
        return "Medium", 5000.0


def infer_continent(country: str) -> str:
    """Infer continent from country name."""
    return COUNTRY_TO_CONTINENT.get(country, "Other")
