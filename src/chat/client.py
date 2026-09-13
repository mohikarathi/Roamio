"""LLM provider abstraction with Gemini API support and reliable offline fallback parser."""

import os
import re
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from src.data.models import UserPreferences, RecommendationResponse
from src.data.normalizer import MONTH_NAMES

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract interface for conversational preference extraction and response generation."""

    @abstractmethod
    def extract_preferences(
        self,
        current_message: str,
        current_preferences: Optional[UserPreferences] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> UserPreferences:
        """Extract or refine structured travel preferences from natural language input."""
        pass

    @abstractmethod
    def format_conversational_response(
        self,
        user_message: str,
        rec_response: RecommendationResponse,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Format an explanation-grounded, conversational response based on recommendation results."""
        pass


COUNTRY_CONTINENT_MAP = {
    "thailand": ("Thailand", "Asia"),
    "thai": ("Thailand", "Asia"),
    "bangkok": ("Thailand", "Asia"),
    "phuket": ("Thailand", "Asia"),
    "chiang mai": ("Thailand", "Asia"),
    "japan": ("Japan", "Asia"),
    "japanese": ("Japan", "Asia"),
    "tokyo": ("Japan", "Asia"),
    "kyoto": ("Japan", "Asia"),
    "indonesia": ("Indonesia", "Asia"),
    "indonesian": ("Indonesia", "Asia"),
    "bali": ("Indonesia", "Asia"),
    "balinese": ("Indonesia", "Asia"),
    "vietnam": ("Vietnam", "Asia"),
    "vietnamese": ("Vietnam", "Asia"),
    "hanoi": ("Vietnam", "Asia"),
    "india": ("India", "Asia"),
    "indian": ("India", "Asia"),
    "france": ("France", "Europe"),
    "french": ("France", "Europe"),
    "paris": ("France", "Europe"),
    "nice": ("France", "Europe"),
    "italy": ("Italy", "Europe"),
    "italian": ("Italy", "Europe"),
    "rome": ("Italy", "Europe"),
    "florence": ("Italy", "Europe"),
    "venice": ("Italy", "Europe"),
    "spain": ("Spain", "Europe"),
    "spanish": ("Spain", "Europe"),
    "barcelona": ("Spain", "Europe"),
    "madrid": ("Spain", "Europe"),
    "greece": ("Greece", "Europe"),
    "greek": ("Greece", "Europe"),
    "athens": ("Greece", "Europe"),
    "santorini": ("Greece", "Europe"),
    "united states": ("United States", "Americas"),
    "usa": ("United States", "Americas"),
    "american": ("United States", "Americas"),
    "peru": ("Peru", "Americas"),
    "peruvian": ("Peru", "Americas"),
    "brazil": ("Brazil", "Americas"),
    "brazilian": ("Brazil", "Americas"),
    "egypt": ("Egypt", "Africa"),
    "egyptian": ("Egypt", "Africa"),
    "cairo": ("Egypt", "Africa"),
    "morocco": ("Morocco", "Africa"),
    "moroccan": ("Morocco", "Africa"),
    "marrakech": ("Morocco", "Africa"),
    "south africa": ("South Africa", "Africa"),
    "south african": ("South Africa", "Africa"),
    "cape town": ("South Africa", "Africa"),
    "tanzania": ("Tanzania", "Africa"),
    "tanzanian": ("Tanzania", "Africa"),
    "zanzibar": ("Tanzania", "Africa"),
    "australia": ("Australia", "Oceania"),
    "australian": ("Australia", "Oceania"),
    "sydney": ("Australia", "Oceania"),
    "melbourne": ("Australia", "Oceania"),
    "new zealand": ("New Zealand", "Oceania"),
    "kiwi": ("New Zealand", "Oceania"),
    "uae": ("United Arab Emirates", "Middle East"),
    "dubai": ("United Arab Emirates", "Middle East"),
    "abu dhabi": ("United Arab Emirates", "Middle East"),
    "turkey": ("Turkey", "Europe"),
    "turkish": ("Turkey", "Europe"),
    "istanbul": ("Turkey", "Europe"),
    "mexico": ("Mexico", "Americas"),
    "mexican": ("Mexico", "Americas"),
    "cancun": ("Mexico", "Americas"),
    "canada": ("Canada", "Americas"),
    "canadian": ("Canada", "Americas"),
    "united kingdom": ("United Kingdom", "Europe"),
    "uk": ("United Kingdom", "Europe"),
    "britain": ("United Kingdom", "Europe"),
    "british": ("United Kingdom", "Europe"),
    "england": ("United Kingdom", "Europe"),
    "english": ("United Kingdom", "Europe"),
    "scotland": ("United Kingdom", "Europe"),
    "scottish": ("United Kingdom", "Europe"),
    "london": ("United Kingdom", "Europe"),
    "edinburgh": ("United Kingdom", "Europe"),
    "iceland": ("Iceland", "Europe"),
    "icelandic": ("Iceland", "Europe"),
    "switzerland": ("Switzerland", "Europe"),
    "swiss": ("Switzerland", "Europe"),
    "zürich": ("Switzerland", "Europe"),
    "zurich": ("Switzerland", "Europe"),
    "geneva": ("Switzerland", "Europe"),
    "portugal": ("Portugal", "Europe"),
    "portuguese": ("Portugal", "Europe"),
    "lisbon": ("Portugal", "Europe"),
    "porto": ("Portugal", "Europe"),
    "germany": ("Germany", "Europe"),
    "german": ("Germany", "Europe"),
    "berlin": ("Germany", "Europe"),
    "munich": ("Germany", "Europe"),
    "netherlands": ("Netherlands", "Europe"),
    "dutch": ("Netherlands", "Europe"),
    "holland": ("Netherlands", "Europe"),
    "amsterdam": ("Netherlands", "Europe"),
    "norway": ("Norway", "Europe"),
    "norwegian": ("Norway", "Europe"),
    "oslo": ("Norway", "Europe"),
    "austria": ("Austria", "Europe"),
    "austrian": ("Austria", "Europe"),
    "vienna": ("Austria", "Europe"),
    "czech republic": ("Czech Republic", "Europe"),
    "czechia": ("Czech Republic", "Europe"),
    "czech": ("Czech Republic", "Europe"),
    "prague": ("Czech Republic", "Europe"),
    "croatia": ("Croatia", "Europe"),
    "croatian": ("Croatia", "Europe"),
    "dubrovnik": ("Croatia", "Europe"),
    "hungary": ("Hungary", "Europe"),
    "hungarian": ("Hungary", "Europe"),
    "budapest": ("Hungary", "Europe"),
    "ireland": ("Ireland", "Europe"),
    "irish": ("Ireland", "Europe"),
    "dublin": ("Ireland", "Europe"),
    "sweden": ("Sweden", "Europe"),
    "swedish": ("Sweden", "Europe"),
    "stockholm": ("Sweden", "Europe"),
    "nepal": ("Nepal", "Asia"),
    "nepalese": ("Nepal", "Asia"),
    "nepali": ("Nepal", "Asia"),
    "sri lanka": ("Sri Lanka", "Asia"),
    "sri lankan": ("Sri Lanka", "Asia"),
    "maldives": ("Maldives", "Asia"),
    "maldivian": ("Maldives", "Asia"),
    "singapore": ("Singapore", "Asia"),
    "singaporean": ("Singapore", "Asia"),
    "malaysia": ("Malaysia", "Asia"),
    "malaysian": ("Malaysia", "Asia"),
    "south korea": ("South Korea", "Asia"),
    "korea": ("South Korea", "Asia"),
    "korean": ("South Korea", "Asia"),
    "seoul": ("South Korea", "Asia"),
}

CURRENCY_TO_INR = {
    "dollar": 85.0,
    "dollars": 85.0,
    "$": 85.0,
    "usd": 85.0,
    "euro": 92.0,
    "euros": 92.0,
    "eur": 92.0,
    "€": 92.0,
    "pound": 110.0,
    "pounds": 110.0,
    "gbp": 110.0,
    "£": 110.0,
    "aud": 55.0,
    "cad": 62.0,
    "sgd": 64.0,
    "thb": 2.4,
    "baht": 2.4,
    "inr": 1.0,
    "rs": 1.0,
    "rs.": 1.0,
    "₹": 1.0,
    "rupees": 1.0,
}


class RuleBasedFallbackClient(BaseLLMClient):
    """
    Deterministic regex and rule-based parser that guarantees 100% functionality
    without requiring an external LLM API key or network connection.
    """

    def extract_preferences(
        self,
        current_message: str,
        current_preferences: Optional[UserPreferences] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> UserPreferences:
        prefs = current_preferences.model_copy() if current_preferences else UserPreferences()
        text = current_message.lower().strip()
        prefs.query_text = current_message

        # 1. Duration extraction: e.g. "8 days", "1 week", "2 weeks", "5 days trip", "for 5 days"
        dur_match = re.search(r"\b(\d+)\s*(?:days?|nights?)\b", text)
        week_match = re.search(r"\b(\d+)\s*weeks?\b", text)
        duration_span = (-1, -1)
        if dur_match:
            prefs.duration_days = int(dur_match.group(1))
            duration_span = dur_match.span()
        elif week_match:
            prefs.duration_days = int(week_match.group(1)) * 7
            duration_span = week_match.span()

        # 2. Budget extraction with multi-currency support
        if "cheaper" in text and prefs.budget_max_inr:
            prefs.budget_max_inr = round(prefs.budget_max_inr * 0.75)
        else:
            budget_extracted = False

            # Mask duration text so numbers in "5 days" aren't confused with budget
            text_masked = text
            if duration_span[0] != -1:
                text_masked = text[:duration_span[0]] + " " * (duration_span[1] - duration_span[0]) + text[duration_span[1]:]

            # Pattern A: explicit currency symbol or unit e.g. "15000 dollars", "$15000", "15k usd"
            curr_units = r"(?:dollars?|usd|\$|euros?|eur|€|pounds?|gbp|£|inr|rs\.?|₹|rupees|baht|thb)"
            pat_a = re.search(
                rf"(?:budget(?:\s+of)?|spend|within|under|max)?\s*({curr_units})?\s*(\d+(?:,\d+)?(?:\.\d+)?)\s*(k|thousand|lakh|lac|m|million)?\s*({curr_units})?",
                text_masked
            )

            # Let's search all matches of numbers in text_masked with surrounding currency / budget keywords
            matches = list(re.finditer(
                rf"(?:(budget(?:\s+of)?|spend|within|under|max)\s*)?({curr_units})?\s*(\d+(?:,\d+)?(?:\.\d+)?)\s*(k|thousand|lakh|lac|m|million)?\s*({curr_units})?",
                text_masked
            ))

            for m in matches:
                keyword, curr_prefix, num_str, mult, curr_suffix = m.groups()
                if not num_str:
                    continue
                # If neither keyword nor currency prefix/suffix is present, skip unless explicitly "budget"
                if not keyword and not curr_prefix and not curr_suffix:
                    continue

                raw_val = float(num_str.replace(",", ""))
                if mult in ["k", "thousand"]:
                    raw_val *= 1000
                elif mult in ["lakh", "lac"]:
                    raw_val *= 100000
                elif mult in ["m", "million"]:
                    raw_val *= 1000000

                # Determine currency multiplier
                c_token = (curr_prefix or curr_suffix or "").lower()
                rate = CURRENCY_TO_INR.get(c_token, 1.0)
                inr_val = raw_val * rate

                if inr_val < 500 and not c_token:
                    inr_val *= 1000  # e.g. "budget 70" -> 70,000

                if inr_val >= 2000:
                    prefs.budget_max_inr = round(inr_val)
                    budget_extracted = True
                    break

            if not budget_extracted:
                # Fallback simple check
                simple_budget = re.search(r"budget\D*?(\d+(?:,\d+)?)", text_masked)
                if simple_budget:
                    v = float(simple_budget.group(1).replace(",", ""))
                    if v < 500:
                        v *= 1000
                    if v >= 2000:
                        prefs.budget_max_inr = round(v)

        # 3. Month extraction: e.g. "in December", "during April"
        for month_name, month_num in MONTH_NAMES.items():
            if re.search(rf"\b{month_name}\b", text):
                prefs.travel_month = month_num
                break

        # 4. Country & Continent extraction with multi-turn intent transition
        # Check for geographic reset intent ("other regions", "different region", "anywhere", "worldwide", "different continent")
        if re.search(r"\b(?:other regions?|different regions?|different continents?|worldwide|anywhere|all regions?|global|globally)\b", text):
            prefs.continents = []
            prefs.countries = []

        new_continents_in_msg: List[str] = []
        new_countries_in_msg: List[str] = []

        # Explicit continent regexes supporting noun and adjective forms
        CONTINENT_PATTERNS = {
            "Europe": r"\b(?:europe|european)\b",
            "Asia": r"\b(?:asia|asian)\b",
            "Africa": r"\b(?:africa|african)\b",
            "Americas": r"\b(?:americas?|american|north america|south america|latin america|central america)\b",
            "Oceania": r"\b(?:oceania|oceanian|australasia|australia|australian|new zealand)\b",
            "Middle East": r"\b(?:middle east|middle eastern)\b",
        }

        for cont_name, cont_regex in CONTINENT_PATTERNS.items():
            if re.search(cont_regex, text):
                if cont_name not in new_continents_in_msg:
                    new_continents_in_msg.append(cont_name)

        for country_key, (canonical_name, cont_name) in COUNTRY_CONTINENT_MAP.items():
            if re.search(rf"\b{country_key}\b", text):
                if canonical_name not in new_countries_in_msg:
                    new_countries_in_msg.append(canonical_name)
                if cont_name not in new_continents_in_msg:
                    new_continents_in_msg.append(cont_name)

        if new_continents_in_msg:
            # User explicitly stated continent(s) in this turn -> replace previous continent(s)
            prefs.continents = new_continents_in_msg
            if new_countries_in_msg:
                prefs.countries = new_countries_in_msg
            else:
                # Keep only previous countries that actually belong to the newly selected continent
                prefs.countries = [
                    c for c in prefs.countries
                    if COUNTRY_CONTINENT_MAP.get(c.lower(), (None, None))[1] in new_continents_in_msg
                ]
        elif new_countries_in_msg:
            # User specified new country/countries in this turn
            prefs.countries = new_countries_in_msg
            derived_continents = []
            for c in new_countries_in_msg:
                c_cont = COUNTRY_CONTINENT_MAP.get(c.lower(), (None, None))[1]
                if c_cont and c_cont not in derived_continents:
                    derived_continents.append(c_cont)
            prefs.continents = derived_continents
        # Else: if no new continent or country was mentioned in current_message, retain previous prefs.continents & prefs.countries

        # 5. Interest and category keywords
        interest_vocab = [
            "beaches", "mountains", "temples", "hiking", "food", "foodie", "seafood",
            "peaceful", "relaxation", "surfing", "diving", "culture", "museums",
            "architecture", "history", "ruins", "adventure", "skiing", "luxury",
            "nature", "waterfalls", "wildlife", "nightlife", "shopping", "snorkeling"
        ]
        extracted_interests = set(prefs.interests)
        for term in interest_vocab:
            if re.search(rf"\b{term}\b", text):
                extracted_interests.add(term)
        prefs.interests = sorted(list(extracted_interests))

        # 6. Dislikes extraction: e.g. "not too much nightlife", "no crowds"
        dislike_match = re.search(r"(?:not too much|no|avoid|without)\s+([a-z\s]+)", text)
        if dislike_match:
            phrase = dislike_match.group(1).strip()
            for token in ["nightlife", "crowds", "parties", "cold", "heat"]:
                if token in phrase and token not in prefs.disliked_features:
                    prefs.disliked_features.append(token)

        return prefs

    def format_conversational_response(
        self,
        user_message: str,
        rec_response: RecommendationResponse,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        items = rec_response.items
        if not items:
            return "I couldn't find destinations matching all those specific criteria. Try expanding your budget or relaxing regional constraints!"

        prefs = rec_response.applied_preferences
        dest_scope = prefs.countries[0] if prefs.countries else (prefs.continents[0] if prefs.continents else "great destinations")
        budget_summary = f"budget within ₹{prefs.budget_max_inr:,.0f}" if prefs.budget_max_inr else "flexible budget"
        duration_summary = f"{prefs.duration_days} days" if prefs.duration_days else "your trip"
        
        lines = []
        lines.append(f"Based on your request to explore **{dest_scope}** for **{duration_summary}** with a **{budget_summary}**, here are my top curated recommendations backed by our hybrid ranking model:\n")

        for item in items:
            dest = item.destination
            reasons_bullets = "\n".join([f"  - {r}" for r in item.explanation.reasons])
            est_trip_cost = dest.est_daily_cost_inr * (prefs.duration_days or 7)
            
            lines.append(
                f"### {item.rank}. **{dest.name}, {dest.country}** *(Match Score: {int(item.final_score * 100)}%)*\n"
                f"- **Overview**: {dest.description}\n"
                f"- **Why Chosen (Model Reasoning)**:\n{reasons_bullets}\n"
                f"- **Trip Cost Estimate**: Approx. ₹{est_trip_cost:,.0f} for {prefs.duration_days or 7} days (₹{dest.est_daily_cost_inr:,.0f}/day) — {item.explanation.budget_fit}.\n"
                f"- **Top Activities**: {', '.join(dest.activities[:4]) if dest.activities else 'Sightseeing, cultural immersion'}\n"
                f"- **Culinary Highlights**: {', '.join(dest.famous_foods[:3]) if dest.famous_foods else 'Local specialties'}\n"
                f"- **Seasonal Timing & Safety**: {', '.join(dest.best_seasons)} (Rating: {dest.safety_rating})\n"
            )

        lines.append("*Tip: You can refine recommendations anytime — for example: 'Focus on beaches', 'Show cheaper options', or ask 'Why was rank 1 selected over rank 2?'*")
        return "\n".join(lines)


class GeminiLLMClient(BaseLLMClient):
    """
    LLM Client powered by Google Gemini API (`google-genai`).
    Uses Gemini for natural language intent extraction and response synthesis,
    strictly adhering to deterministic recommendation features.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.fallback = RuleBasedFallbackClient()
        self.client = None
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info("GeminiLLMClient initialized successfully.")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini Client: {e}. Falling back to RuleBasedFallbackClient.")

    def extract_preferences(
        self,
        current_message: str,
        current_preferences: Optional[UserPreferences] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> UserPreferences:
        if not self.client:
            return self.fallback.extract_preferences(current_message, current_preferences, conversation_history)

        try:
            curr_dict = current_preferences.model_dump() if current_preferences else {}
            prompt = f"""
            You are Roamio's conversational intent parser.
            Convert the traveler's message into a structured JSON travel preference object.
            
            Current cumulative preferences:
            {json.dumps(curr_dict, indent=2)}

            User's new message:
            "{current_message}"

            CRITICAL GEOGRAPHIC RULES:
            - If the user changes or specifies their geographic focus (e.g., from Asia to Europe, or mentions a new country/continent), update continents and countries to reflect the new focus rather than keeping previous mutually exclusive locations.
            - If the user asks for other regions, different regions, anywhere, or worldwide, reset continents and countries to empty lists [].

            Return ONLY valid JSON matching this schema:
            {{
              "query_text": string,
              "budget_max_inr": float or null,
              "duration_days": int or null,
              "travel_month": int (1-12) or null,
              "continents": list of strings,
              "countries": list of strings,
              "categories": list of strings,
              "interests": list of strings,
              "disliked_features": list of strings
            }}
            """
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            # Parse JSON from response
            raw_text = response.text.strip()
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group(0))
                # Validate with Pydantic
                return UserPreferences(**extracted)
        except Exception as e:
            logger.warning(f"Gemini preference extraction failed: {e}. Using rule fallback.")

        return self.fallback.extract_preferences(current_message, current_preferences, conversation_history)

    def format_conversational_response(
        self,
        user_message: str,
        rec_response: RecommendationResponse,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        if not self.client:
            return self.fallback.format_conversational_response(user_message, rec_response, conversation_history)

        try:
            # Prepare grounded context for the LLM
            items_payload = []
            for item in rec_response.items:
                items_payload.append({
                    "rank": item.rank,
                    "name": item.destination.name,
                    "country": item.destination.country,
                    "category": item.destination.category,
                    "cost_per_day_inr": item.destination.est_daily_cost_inr,
                    "safety": item.destination.safety_rating,
                    "score": round(item.final_score * 100, 1),
                    "reasons": item.explanation.reasons
                })

            system_instruction = """
            You are Roamio, an expert AI travel recommendation concierge.
            CRITICAL RULES:
            1. You MUST NOT invent or recommend any destination outside the provided recommendation payload.
            2. Ground all answers and explanations directly on the provided reasons, costs, and scores.
            3. Present the ranked destinations clearly with markdown formatting, mentioning why each fits the traveler's goals.
            4. Keep your tone enthusiastic, helpful, and concise.
            """

            prompt = f"""
            User inquiry: "{user_message}"
            Ranked Recommendations from Roamio engine:
            {json.dumps(items_payload, indent=2)}

            Please deliver a polished conversational response to the traveler.
            """

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"system_instruction": system_instruction}
            )
            return response.text
        except Exception as e:
            logger.warning(f"Gemini response formatting failed: {e}. Using rule fallback.")
            return self.fallback.format_conversational_response(user_message, rec_response, conversation_history)


def get_llm_client() -> BaseLLMClient:
    """Factory function returning GeminiLLMClient if API key exists, otherwise RuleBasedFallbackClient."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return GeminiLLMClient(api_key)
    return RuleBasedFallbackClient()
