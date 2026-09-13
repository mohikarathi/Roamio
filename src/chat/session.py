"""Conversational state manager handling multi-turn travel preferences, refinements, and comparisons."""

import time
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

from src.data.models import UserPreferences, RecommendationResponse, RecommendationItem
from src.ranking.engine import RecommendationEngine
from src.chat.client import BaseLLMClient, get_llm_client


class ChatSession:
    """
    Manages multi-turn conversational state, incremental preference refinement,
    and recommendation execution.
    """

    def __init__(
        self,
        engine: Optional[RecommendationEngine] = None,
        llm_client: Optional[BaseLLMClient] = None
    ):
        self.engine = engine or RecommendationEngine()
        self.llm_client = llm_client or get_llm_client()
        self.history: List[Dict[str, str]] = []
        self.preferences = UserPreferences()
        self.last_response: Optional[RecommendationResponse] = None
        self.alternatives_offset = 0

    def reset(self) -> None:
        """Reset conversational state and preferences."""
        self.history = []
        self.preferences = UserPreferences()
        self.last_response = None
        self.alternatives_offset = 0

    def _handle_comparison_query(self, text: str) -> Optional[str]:
        """Check if user is asking to compare destinations from the previous turn."""
        if not self.last_response or not self.last_response.items:
            return None

        items = self.last_response.items
        lower = text.lower()

        # Check queries like "compare 1 and 2", "compare first and second", "compare first and third"
        is_compare = "compare" in lower
        if not is_compare:
            return None

        idx1, idx2 = 0, 1
        if "first and third" in lower or "1 and 3" in lower:
            idx1, idx2 = 0, min(2, len(items) - 1)
        elif "second and third" in lower or "2 and 3" in lower:
            idx1, idx2 = 1, min(2, len(items) - 1)

        d1 = items[idx1].destination
        d2 = items[idx2].destination

        table = (
            f"### Comparison: **{d1.name}** vs **{d2.name}**\n\n"
            f"| Feature | {d1.name} ({d1.country}) | {d2.name} ({d2.country}) |\n"
            f"| :--- | :--- | :--- |\n"
            f"| **Category** | {d1.category} | {d2.category} |\n"
            f"| **Est. Daily Cost** | ₹{d1.est_daily_cost_inr:,.0f} | ₹{d2.est_daily_cost_inr:,.0f} |\n"
            f"| **Safety Rating** | {d1.safety_rating} | {d2.safety_rating} |\n"
            f"| **Best Seasons** | {', '.join(d1.best_seasons or ['All'])} | {', '.join(d2.best_seasons or ['All'])} |\n"
            f"| **Famous Foods** | {', '.join(d1.famous_foods[:3])} | {', '.join(d2.famous_foods[:3])} |\n"
            f"| **Match Score** | {int(items[idx1].final_score * 100)}% | {int(items[idx2].final_score * 100)}% |\n"
            f"| **Key Reason** | {items[idx1].explanation.reasons[0]} | {items[idx2].explanation.reasons[0]} |\n\n"
            f"**Recommendation**: Choose **{d1.name}** for {d1.category.lower()} ambiance or **{d2.name}** if you prefer {d2.cost_level.lower()} budget travel."
        )
        return table

    def _handle_why_query(self, text: str) -> Optional[str]:
        """Check if user is asking why a destination was ranked or why X is above Y."""
        if not self.last_response or not self.last_response.items:
            return None

        lower = text.lower()
        if not lower.startswith("why") and "why did you rank" not in lower:
            return None

        items = self.last_response.items
        top_dest = items[0].destination
        exp = items[0].explanation

        reasons_str = "\n".join([f"- **{r}**" for r in exp.reasons])
        breakdown_str = "\n".join([f"- {k}: {v * 100:.1f}%" for k, v in exp.feature_contributions.items()])

        return (
            f"### Why Roamio Ranked **{top_dest.name}** at #1:\n\n"
            f"The ranking decision is based on verified multi-signal scoring, not guesswork:\n\n"
            f"**Core Evidence Signals**:\n{reasons_str}\n\n"
            f"**Underlying Ranking Contributions**:\n{breakdown_str}\n\n"
            f"*{top_dest.name} scored a composite {int(items[0].final_score * 100)}% match, outperforming other candidates on both semantic alignment and budget compatibility.*"
        )

    def process_turn(self, user_message: str) -> Tuple[str, Optional[RecommendationResponse]]:
        """
        Process a user conversation turn:
        1. Checks for specific intent (comparison, ranking explanation, alternatives).
        2. Refines cumulative preference object.
        3. Calls RecommendationEngine.
        4. Formats conversational response.
        """
        self.history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # 1. Check for explanation query ("Why did you rank X first?")
        why_answer = self._handle_why_query(user_message)
        if why_answer:
            self.history.append({"role": "assistant", "content": why_answer, "timestamp": datetime.now(timezone.utc).isoformat()})
            return why_answer, self.last_response

        # 2. Check for comparison query ("Compare 1 and 2")
        comparison_answer = self._handle_comparison_query(user_message)
        if comparison_answer:
            self.history.append({"role": "assistant", "content": comparison_answer, "timestamp": datetime.now(timezone.utc).isoformat()})
            return comparison_answer, self.last_response

        # 3. Check for alternatives query ("Show alternatives")
        if "alternative" in user_message.lower():
            self.alternatives_offset += 5
        else:
            self.alternatives_offset = 0

        # 4. Extract & refine structured preferences
        self.preferences = self.llm_client.extract_preferences(
            current_message=user_message,
            current_preferences=self.preferences,
            conversation_history=self.history
        )

        # 5. Run recommendation engine
        try:
            rec_response = self.engine.recommend(
                preferences=self.preferences,
                top_k=5 + self.alternatives_offset,
                apply_diversity=True
            )
        except Exception:
            self.engine = RecommendationEngine()
            rec_response = self.engine.recommend(
                preferences=self.preferences,
                top_k=5 + self.alternatives_offset,
                apply_diversity=True
            )

        # Offset items if alternatives requested
        if self.alternatives_offset > 0 and len(rec_response.items) > self.alternatives_offset:
            rec_response.items = rec_response.items[self.alternatives_offset:self.alternatives_offset + 5]
            for idx, it in enumerate(rec_response.items, start=1):
                it.rank = idx

        self.last_response = rec_response

        # 6. Generate grounded response
        assistant_reply = self.llm_client.format_conversational_response(
            user_message=user_message,
            rec_response=rec_response,
            conversation_history=self.history
        )

        self.history.append({
            "role": "assistant",
            "content": assistant_reply,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        return assistant_reply, rec_response
