"""Unit tests for chat preference extraction and multi-turn session management."""

import pytest
from src.data.models import UserPreferences
from src.chat.client import RuleBasedFallbackClient
from src.chat.session import ChatSession


def test_fallback_preference_extraction():
    client = RuleBasedFallbackClient()
    message = "I have ₹70,000, 8 days, and I want somewhere warm in December with beaches, good food and not too much nightlife."
    prefs = client.extract_preferences(message)

    assert prefs.budget_max_inr == 70000.0
    assert prefs.duration_days == 8
    assert prefs.travel_month == 12
    assert "beaches" in prefs.interests
    assert "food" in prefs.interests
    assert "nightlife" in prefs.disliked_features


def test_chat_session_refinement_flow():
    session = ChatSession()

    # Turn 1: Initial query
    reply1, rec1 = session.process_turn("I want a trip to Asia for ₹80,000 and 10 days with temples.")
    assert len(session.history) == 2
    assert session.preferences.budget_max_inr == 80000.0
    assert "Asia" in session.preferences.continents
    assert rec1 is not None and len(rec1.items) > 0

    # Turn 2: Refinement ("cheaper")
    reply2, rec2 = session.process_turn("Actually, make it cheaper.")
    assert session.preferences.budget_max_inr == 60000.0  # reduced by 25%
    assert rec2 is not None

    # Turn 3: Explanation question
    reply3, _ = session.process_turn("Why did you rank the first place so high?")
    assert "Why Roamio Ranked" in reply3
    assert "Core Evidence Signals" in reply3

    # Turn 4: Comparison question
    reply4, _ = session.process_turn("Compare 1 and 2.")
    assert "Comparison:" in reply4
    assert "Est. Daily Cost" in reply4


def test_country_and_currency_extraction():
    client = RuleBasedFallbackClient()
    message = "i want to travel to thailand for 5 days on a budget of 15000 dollars, suggest some places i can visit there in the mean time"
    prefs = client.extract_preferences(message)

    assert prefs.countries == ["Thailand"]
    assert "Asia" in prefs.continents
    assert prefs.duration_days == 5
    assert prefs.budget_max_inr == 15000 * 85  # 1,275,000 INR


def test_thailand_recommendation_flow():
    session = ChatSession()
    message = "i want to travel to thailand for 5 days on a budget of 15000 dollars, suggest some places i can visit there in the mean time"
    reply, rec = session.process_turn(message)

    assert rec is not None
    assert len(rec.items) >= 4
    for it in rec.items:
        assert it.destination.country == "Thailand"
        assert len(it.explanation.reasons) > 0
    assert "Thailand" in reply
    assert "Model Reasoning" in reply or "Why Chosen" in reply


def test_multi_turn_asia_to_europe_switch():
    """Verify that switching from Asia to Europe completely overrides Asia filters and never suggests Bali for Europe."""
    session = ChatSession()

    # Turn 1: Asia beach query
    turn1_msg = "I have a budget of ₹70,000 for 8 days. I want a warm beach destination in Asia with great food and relaxed vibes."
    _, rec1 = session.process_turn(turn1_msg)
    assert rec1 is not None and len(rec1.items) > 0
    assert "Asia" in session.preferences.continents
    turn1_names = [it.destination.name.lower() for it in rec1.items]
    assert any("bali" in name or "hua hin" in name for name in turn1_names)

    # Turn 2: Switch to Europe trips
    turn2_msg = "Now suggest Europe trips"
    reply2, rec2 = session.process_turn(turn2_msg)
    assert rec2 is not None and len(rec2.items) > 0
    # Continent must be Europe only (not Asia)
    assert session.preferences.continents == ["Europe"]
    assert "Asia" not in session.preferences.continents

    # Critical check: Bali or any non-European destination MUST NOT appear
    turn2_names = [it.destination.name.lower() for it in rec2.items]
    assert not any("bali" in name for name in turn2_names)
    for it in rec2.items:
        assert it.destination.continent == "Europe", f"Expected Europe but got {it.destination.name} ({it.destination.continent})"


def test_adjective_continent_and_country_extraction():
    """Verify adjective forms (European, French, Italian) and regional resets."""
    client = RuleBasedFallbackClient()

    # European adjective
    p1 = client.extract_preferences("Recommend historic European cities known for museums and art.")
    assert p1.continents == ["Europe"]

    # French adjective and nationality
    p2 = client.extract_preferences("Show me French coastal towns and countryside.")
    assert "France" in p2.countries
    assert p2.continents == ["Europe"]

    # Italian adjective
    p3 = client.extract_preferences("Italian food and ancient ruins tour.")
    assert "Italy" in p3.countries
    assert p3.continents == ["Europe"]

    # Region reset
    p4 = client.extract_preferences(
        "Show me alternative destinations in other regions",
        current_preferences=UserPreferences(continents=["Europe"], countries=["France"])
    )
    assert p4.continents == []
    assert p4.countries == []


