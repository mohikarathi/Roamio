"""Roamio 2.0 - Conversational Hybrid Travel Recommendation System.

Modern, editorial travel discovery application with conversational AI concierge,
two-stage hybrid retrieval, MMR diversity, and evidence-based explainability.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

from src.config import DEFAULT_WEIGHTS
from src.data.models import UserPreferences, RecommendationResponse, Destination, RecommendationItem
from src.data.db import get_all_destinations, get_destination_by_id, get_database_stats
from src.data.images.local_cache import get_image_data_uri, get_image_web_url
from src.data.images.gallery import get_destination_gallery
from src.ranking.engine import RecommendationEngine
from src.chat.session import ChatSession
from src.evaluation.runner import run_ablation_study

# Streamlit Page Setup
st.set_page_config(
    page_title="Roamio — Conversational Travel Discovery",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Design System: Calm, warm, premium editorial travel aesthetic
st.markdown("""
<style>
    /* Global Canvas */
    .stApp {
        background-color: #F7F5F0 !important;
        color: #172B3A !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1240px !important;
    }

    /* Headings and Base Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #172B3A !important;
        font-weight: 600 !important;
        letter-spacing: -0.015em !important;
    }
    p, span, label, div {
        color: #172B3A;
    }
    hr {
        border: none !important;
        border-top: 1px solid #E4E0D8 !important;
        margin: 1.5rem 0 !important;
    }

    /* Force Light Mode on Streamlit Inputs & Selectboxes */
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stChatInput"] textarea {
        background-color: #FFFFFF !important;
        color: #172B3A !important;
        border: 1px solid #E4E0D8 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within,
    div[data-testid="stChatInput"] textarea:focus {
        border-color: #C96F4A !important;
        box-shadow: 0 0 0 1px #C96F4A !important;
    }
    div[data-baseweb="select"] span {
        color: #172B3A !important;
    }
    
    /* Dropdown Menus */
    ul[data-baseweb="menu"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E4E0D8 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(23, 43, 58, 0.08) !important;
    }
    li[data-baseweb="menu-item"] {
        color: #172B3A !important;
    }
    li[data-baseweb="menu-item"]:hover,
    li[data-baseweb="menu-item"][aria-selected="true"] {
        background-color: #F7F5F0 !important;
        color: #C96F4A !important;
    }

    /* Chat Messages */
    div[data-testid="stChatMessage"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E4E0D8 !important;
        border-radius: 12px !important;
        padding: 1.1rem 1.3rem !important;
        margin-bottom: 0.9rem !important;
        box-shadow: 0 1px 3px rgba(23, 43, 58, 0.03) !important;
    }
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li {
        color: #172B3A !important;
        line-height: 1.6 !important;
    }
    /* User Message Bubble with Soft Peach Accent */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background-color: #F3DED3 !important;
        border: 1px solid #E4E0D8 !important;
    }
    
    /* Expanders */
    div[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E4E0D8 !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 3px rgba(23, 43, 58, 0.03) !important;
        margin-top: -0.5rem !important;
        margin-bottom: 1.25rem !important;
    }
    div[data-testid="stExpander"] details summary {
        color: #172B3A !important;
        font-weight: 600 !important;
    }
    div[data-testid="stExpander"] details summary:hover {
        color: #C96F4A !important;
    }
    div[data-testid="stExpander"] details summary svg {
        fill: #66737D !important;
    }

    /* Premium Elevated Navigation Header */
    .nav-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #FFFFFF;
        border: 1px solid #E4E0D8;
        border-radius: 14px;
        padding: 0.85rem 1.4rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 4px rgba(23, 43, 58, 0.03);
    }
    .brand-logo-wrap {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-badge {
        width: 38px;
        height: 38px;
        background: #F3DED3;
        border: 1px solid rgba(201, 111, 74, 0.25);
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .brand-logo {
        font-size: 1.68rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #172B3A;
        margin: 0;
        line-height: 1;
    }
    .brand-edition-chip {
        background: #F7F5F0;
        border: 1px solid #E4E0D8;
        color: #66737D;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 4px 9px;
        border-radius: 6px;
        margin-left: 4px;
    }
    .nav-pills-wrap {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .nav-pill {
        background: #F7F5F0;
        border: 1px solid #E4E0D8;
        color: #172B3A;
        font-size: 0.82rem;
        font-weight: 500;
        padding: 5px 12px;
        border-radius: 20px;
        display: flex;
        align-items: center;
        gap: 7px;
    }
    .nav-pill-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #55745A;
        display: inline-block;
    }
    .nav-pill-accent {
        background: #F3DED3;
        border: 1px solid rgba(201, 111, 74, 0.3);
        color: #C96F4A;
        font-weight: 600;
    }

    /* Tabs Styling - Editorial & Warm */
    div[data-baseweb="tab-list"] {
        gap: 1.75rem !important;
        border-bottom: 1px solid #E4E0D8 !important;
        margin-bottom: 1.5rem !important;
    }
    button[data-baseweb="tab"] {
        font-size: 1.02rem !important;
        font-weight: 500 !important;
        color: #66737D !important;
        padding: 0.65rem 0.5rem !important;
        border-bottom: 2px solid transparent !important;
        transition: color 0.2s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #C96F4A !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #172B3A !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #C96F4A !important;
    }

    /* Modern Travel Destination Card */
    .travel-card {
        background: #FFFFFF;
        border: 1px solid #E4E0D8;
        border-radius: 12px 12px 0 0;
        overflow: hidden;
        margin-bottom: 0;
        box-shadow: 0 1px 4px rgba(23, 43, 58, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .travel-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(23, 43, 58, 0.07);
    }
    /* Multi-Image Gallery Strip (3-4 photos filling the rectangle) */
    .card-gallery-strip {
        display: grid;
        grid-template-columns: 2.2fr 1fr 1fr 1fr;
        height: 180px;
        gap: 2px;
        background-color: #E9E6DE;
        overflow: hidden;
    }
    .gallery-tile {
        position: relative;
        width: 100%;
        height: 100%;
        overflow: hidden;
        background-color: #E9E6DE;
    }
    .gallery-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
        transition: transform 0.25s ease;
    }
    .gallery-img:hover {
        transform: scale(1.04);
    }
    .gallery-tile-author {
        position: absolute;
        bottom: 4px;
        right: 5px;
        background: rgba(23, 43, 58, 0.75);
        color: #F7F5F0;
        padding: 1px 5px;
        border-radius: 3px;
        font-size: 0.62rem;
        pointer-events: none;
        white-space: nowrap;
        max-width: 90%;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .card-content {
        padding: 1.1rem 1.25rem;
    }
    .card-title-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 0.4rem;
    }
    .card-title {
        font-size: 1.18rem;
        font-weight: 600;
        color: #172B3A;
        margin: 0;
    }
    .card-rank {
        font-size: 0.8rem;
        font-weight: 600;
        color: #172B3A;
        background: #F7F5F0;
        border: 1px solid #E4E0D8;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .card-meta-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-bottom: 0.65rem;
    }
    .meta-chip {
        background-color: #F7F5F0;
        color: #66737D;
        border: 1px solid #E4E0D8;
        padding: 3px 8px;
        border-radius: 5px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .meta-chip-cost {
        background-color: #F3DED3;
        color: #C96F4A;
        border: 1px solid #E4E0D8;
        padding: 3px 8px;
        border-radius: 5px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .meta-chip-safety-high {
        background-color: #EDF3EE;
        color: #55745A;
        border: 1px solid #D5E2D6;
        padding: 3px 8px;
        border-radius: 5px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .meta-chip-safety-med {
        background-color: #F7F5F0;
        color: #66737D;
        border: 1px solid #E4E0D8;
        padding: 3px 8px;
        border-radius: 5px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .card-snippet {
        font-size: 0.88rem;
        color: #66737D;
        line-height: 1.5;
        margin: 0.5rem 0 0.65rem 0;
    }
    .card-rationale {
        font-size: 0.82rem;
        color: #172B3A;
        background-color: #F7F5F0;
        border-left: 3px solid #C96F4A;
        border-top: 1px solid #E4E0D8;
        border-right: 1px solid #E4E0D8;
        border-bottom: 1px solid #E4E0D8;
        padding: 6px 10px;
        border-radius: 0 4px 4px 0;
        margin-top: 0.5rem;
        line-height: 1.4;
    }

    /* Detail View Styling */
    .detail-container {
        background: #FFFFFF;
        border: 1px solid #E4E0D8;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(23, 43, 58, 0.04);
    }
    .detail-hero-box {
        position: relative;
        width: 100%;
        height: 320px;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 1.25rem;
        border: 1px solid #E4E0D8;
    }
    .detail-hero-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    /* Folium Map Container Integration - Seamlessly fills container rectangle */
    div[data-testid="stCustomComponentV1"] {
        width: 100% !important;
        display: flex !important;
    }
    div[data-testid="stCustomComponentV1"] > iframe,
    iframe[title*="folium"] {
        width: 100% !important;
        min-width: 100% !important;
        border: 1px solid #E4E0D8 !important;
        border-radius: 12px !important;
        background-color: #E9E6DE !important;
        box-shadow: 0 1px 4px rgba(23, 43, 58, 0.05) !important;
        display: block !important;
    }

    /* Buttons: Primary Terracotta, Secondary White with Warm Gray border */
    div.stButton > button[kind="primary"],
    div.stButton > button[type="primary"] {
        background-color: #C96F4A !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        transition: background-color 0.2s ease !important;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[type="primary"]:hover {
        background-color: #B35E3B !important;
        color: #FFFFFF !important;
    }
    div.stButton > button {
        background-color: #FFFFFF !important;
        color: #172B3A !important;
        border: 1px solid #E4E0D8 !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        background-color: #F7F5F0 !important;
        border-color: #C96F4A !important;
        color: #C96F4A !important;
    }

    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #172B3A !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #66737D !important;
    }

    /* Dataframe / Table in Architecture */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E4E0D8 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_recommendation_engine():
    """Cache recommendation engine instance across user sessions."""
    return RecommendationEngine()


@st.cache_data
def get_cached_db_stats():
    """Cache database catalog metrics."""
    return get_database_stats()


@st.cache_data
def get_cached_destinations():
    """Cache full destinations catalog for spatial exploration."""
    return get_all_destinations()


# Initialize state
engine = get_recommendation_engine()
db_stats = get_cached_db_stats()
all_destinations = get_cached_destinations()

if "chat_session" not in st.session_state:
    st.session_state["chat_session"] = ChatSession(engine=engine)
if "explore_response" not in st.session_state:
    st.session_state["explore_response"] = None
if "selected_dest_id" not in st.session_state:
    st.session_state["selected_dest_id"] = None

chat_session: ChatSession = st.session_state["chat_session"]

# Top Navigation Bar with SVG branding (Deep Navy & Terracotta accent)
st.markdown("""
<div class="nav-header">
    <div class="brand-logo-wrap">
        <div class="brand-badge">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#C96F4A" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="12 2 19 21 12 17 5 21 12 2"/>
            </svg>
        </div>
        <div class="brand-logo">Roamio<span style="color:#C96F4A;">.</span></div>
        <span class="brand-edition-chip">Curated World Guide</span>
    </div>
    <div class="nav-pills-wrap">
        <div class="nav-pill">
            <span class="nav-pill-dot"></span>
            <span>252 Destinations</span>
        </div>
        <div class="nav-pill nav-pill-accent">
            <span>AI Concierge</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


def render_folium_map(dest_list, active_id=None, height=480, zoom_start=2):
    """Render muted architectural travel Folium map with markers and photo popups."""
    if not dest_list:
        return None

    valid_dests = [d for d in dest_list if d.latitude != 0.0 and d.longitude != 0.0]
    if not valid_dests:
        return None

    if active_id:
        active_match = [d for d in valid_dests if d.destination_id == active_id]
        if active_match:
            center_lat, center_lon = active_match[0].latitude, active_match[0].longitude
            zoom_start = 5
        else:
            center_lat = np.mean([d.latitude for d in valid_dests])
            center_lon = np.mean([d.longitude for d in valid_dests])
    else:
        center_lat = np.mean([d.latitude for d in valid_dests])
        center_lon = np.mean([d.longitude for d in valid_dests])

    # Esri World Topo Map: Clean, high-resolution, travel-editorial cartography
    # Warm ivory land tones, soft cyan water, relief shading, and zero watermarks/API keys
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, USGS, FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey",
        control_scale=False,
        width="100%",
        height="100%"
    )

    for d in valid_dests:
        is_active = (d.destination_id == active_id)
        img_web = get_image_web_url(d.thumbnail_url or d.image_url)
        img_tag = f'<img src="{img_web}" style="width:100%; height:110px; object-fit:cover; border-radius:6px; margin-bottom:8px; border:1px solid #E4E0D8;" />' if img_web else ""
        seasons_info = ', '.join(d.best_seasons[:2]) if d.best_seasons else 'Year-round'

        desc_text = d.description or ""
        desc_clean = desc_text.replace('"', '&quot;').replace("'", "&#39;")
        if len(desc_clean) > 240:
            desc_clean = desc_clean[:237] + "..."

        activities_list = d.activities[:3] if d.activities else ["Sightseeing", "Local culture"]
        activities_text = ", ".join(activities_list)

        popup_html = (
            f'<div style="width:280px; font-family:-apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif; color:#172B3A; padding:4px;">'
            f'{img_tag}'
            f'<div style="font-weight:700; font-size:14.5px; color:#172B3A; margin-bottom:3px;">{d.name}, {d.country}</div>'
            f'<div style="display:flex; flex-wrap:wrap; gap:4px; margin-bottom:6px; font-size:11px;">'
            f'<span style="background:#F3DED3; color:#C96F4A; font-weight:600; padding:1px 6px; border-radius:3px;">{d.category}</span>'
            f'<span style="background:#F7F5F0; border:1px solid #E4E0D8; color:#172B3A; font-weight:600; padding:1px 6px; border-radius:3px;">₹{d.est_daily_cost_inr:,.0f}/day</span>'
            f'<span style="background:#EBF2EC; color:#55745A; font-weight:600; padding:1px 6px; border-radius:3px;">{d.safety_rating} Safety</span>'
            f'</div>'
            f'<div style="font-size:11.5px; line-height:1.45; color:#3A4D59; margin-bottom:6px; max-height:85px; overflow-y:auto; border-left:2px solid #C96F4A; padding-left:6px;">'
            f'{desc_clean}'
            f'</div>'
            f'<div style="font-size:10.5px; color:#66737D; border-top:1px solid #E4E0D8; padding-top:4px;">'
            f'<div><strong>Top Activities</strong>: {activities_text}</div>'
            f'<div><strong>Best Time</strong>: {seasons_info}</div>'
            f'</div>'
            f'</div>'
        )

        # Elegant SVG Teardrop Pin: Normal = Terracotta (#C96F4A), Selected = Deep Navy (#172B3A)
        pin_bg = "#172B3A" if is_active else "#C96F4A"
        dot_bg = "#FFFFFF"
        pin_w = 26 if is_active else 20
        pin_h = 34 if is_active else 27
        z_index = 1000 if is_active else 100

        svg_html = (
            f'<div style="z-index:{z_index}; cursor:pointer; width:{pin_w}px; height:{pin_h}px;">'
            f'<svg width="{pin_w}" height="{pin_h}" viewBox="0 0 24 32" fill="none" style="filter: drop-shadow(0 2px 4px rgba(23,43,58,0.32));">'
            f'<path d="M12 0C5.37 0 0 5.37 0 12c0 9 12 20 12 20s12-11 12-20c0-6.63-5.37-12-12-12z" fill="{pin_bg}" stroke="#FFFFFF" stroke-width="2"/>'
            f'<circle cx="12" cy="11" r="4" fill="{dot_bg}"/>'
            f'</svg>'
            f'</div>'
        )

        custom_icon = folium.DivIcon(
            html=svg_html,
            icon_size=(pin_w, pin_h),
            icon_anchor=(pin_w // 2, pin_h)
        )

        folium.Marker(
            location=[d.latitude, d.longitude],
            tooltip=f"{d.name}, {d.country}",
            popup=folium.Popup(popup_html, max_width=300),
            icon=custom_icon
        ).add_to(m)

    # Custom styling for Leaflet controls, canvas background, and popups
    custom_map_css = """
    <style>
        .leaflet-container {
            background-color: #E9E6DE !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        }
        .leaflet-bar a {
            background-color: #FFFFFF !important;
            color: #172B3A !important;
            border-bottom: 1px solid #E4E0D8 !important;
        }
        .leaflet-bar a:hover {
            background-color: #F7F5F0 !important;
            color: #C96F4A !important;
        }
        .leaflet-popup-content-wrapper {
            background: #FFFFFF !important;
            border: 1px solid #E4E0D8 !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 14px rgba(23, 43, 58, 0.12) !important;
            padding: 4px !important;
        }
        .leaflet-popup-tip {
            background: #FFFFFF !important;
            border: 1px solid #E4E0D8 !important;
        }
        .leaflet-control-attribution {
            background: rgba(247, 245, 240, 0.85) !important;
            color: #66737D !important;
            font-size: 9px !important;
        }
        .leaflet-control-attribution a {
            color: #66737D !important;
        }
    </style>
    <script>
        setTimeout(function() {
            window.dispatchEvent(new Event('resize'));
        }, 150);
    </script>
    """
    m.get_root().html.add_child(folium.Element(custom_map_css))

    return m


def render_destination_detail_modal(dest: Destination, explanation=None):
    """Editorial destination detail view with photography and activity guide."""
    with st.container():
        st.markdown("""<div class="detail-container">""", unsafe_allow_html=True)
        
        img_web = get_image_web_url(dest.thumbnail_url or dest.image_url)
        img_fallback = get_image_data_uri(dest.thumbnail_url or dest.image_url)
        author_text = dest.photo_author or "Unsplash Contributor"
        author_link = dest.photo_author_url or "https://unsplash.com"
        
        modal_html = (
            f'<div class="detail-hero-box">'
            f'<img class="detail-hero-img" src="{img_web}" onerror="this.onerror=null; this.src=\'{img_fallback}\';" alt="{dest.name}" />'
            f'<div class="photo-credit">Photo by <a href="{author_link}" target="_blank">{author_text}</a> on {dest.image_provider or "Unsplash"}</div>'
            f'</div>'
            f'<h2 style="margin:0 0 0.5rem 0; color:#172B3A;">{dest.name}, {dest.country}</h2>'
            f'<div style="color:#66737D; font-size:0.95rem; margin-bottom:1rem;">'
            f'<b>Region:</b> {dest.region or "Scenic"} ({dest.continent}) &nbsp;|&nbsp; '
            f'<b>Category:</b> {dest.category} &nbsp;|&nbsp; '
            f'<b>Daily Cost:</b> ₹{dest.est_daily_cost_inr:,.0f} ({dest.cost_level}) &nbsp;|&nbsp; '
            f'<b>Safety:</b> {dest.safety_rating} Safety'
            f'</div>'
            f'<p style="color:#172B3A; font-size:1.02rem; line-height:1.6;">{dest.description}</p>'
        )
        st.markdown(modal_html, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if dest.cultural_significance:
                st.markdown(f"**Heritage & Culture**: {dest.cultural_significance}")
            if dest.activities:
                st.markdown(f"**Recommended Activities**: {', '.join(dest.activities)}")
        with col_b:
            if dest.famous_foods:
                st.markdown(f"**Signature Local Cuisine**: {', '.join(dest.famous_foods)}")
            if dest.best_seasons:
                st.markdown(f"**Optimal Travel Seasons**: {', '.join(dest.best_seasons)}")

        if explanation:
            with st.expander("Ranking Evidence & Feature Breakdown"):
                st.markdown("**Deterministic Feature Signals**:")
                for r in explanation.reasons:
                    st.markdown(f"- {r}")
                
                st.markdown("**Relative Signal Contributions**:")
                f_cols = st.columns(len(explanation.feature_contributions))
                for idx, (k, v) in enumerate(explanation.feature_contributions.items()):
                    with f_cols[idx]:
                        st.metric(label=k, value=f"{v * 100:.0f}%")
        
        st.markdown("</div>", unsafe_allow_html=True)


def render_destination_card(
    item: RecommendationItem,
    key_prefix: str = "chat",
    duration_days: int = 7,
    show_map_button: bool = False
):
    """Render modern editorial destination card with 4-photo gallery strip and expandable overview & reasoning."""
    d = item.destination
    gallery = get_destination_gallery(d)

    tiles = []
    for g in gallery[:4]:
        author = g.get("author", "Unsplash")
        alt = g.get("alt", f"{d.name}, {d.country}")
        web_uri = g.get("web_url") or g.get("data_uri", "")
        fallback_uri = g.get("data_uri", "")
        tile_html = (
            f'<div class="gallery-tile">'
            f'<img class="gallery-img" src="{web_uri}" onerror="this.onerror=null; this.src=\'{fallback_uri}\';" alt="{alt}" title="{alt}" />'
            f'<div class="gallery-tile-author">{author}</div>'
            f'</div>'
        )
        tiles.append(tile_html)
    gallery_tiles_html = "".join(tiles)

    badge_budget = f"₹{d.est_daily_cost_inr:,.0f} / day"
    est_trip_cost = d.est_daily_cost_inr * (duration_days or 7)
    reasons_list = item.explanation.reasons or ["Strong semantic and contextual match."]
    top_reason = reasons_list[0]
    safety_chip_class = "meta-chip-safety-high" if (d.safety_rating and d.safety_rating.lower() == "high") else "meta-chip"
    seasons_text = f"Best: {', '.join(d.best_seasons[:2])}" if d.best_seasons else "Best: Year-round"

    card_html = (
        f'<div class="travel-card">'
        f'<div class="card-gallery-strip">{gallery_tiles_html}</div>'
        f'<div class="card-content">'
        f'<div class="card-title-row">'
        f'<h3 class="card-title">{d.name}, {d.country}</h3>'
        f'<span class="card-rank">Pick #{item.rank} · {int(item.final_score * 100)}% Match</span>'
        f'</div>'
        f'<div class="card-meta-chips">'
        f'<span class="meta-chip">{d.category}</span>'
        f'<span class="meta-chip-cost">{badge_budget}</span>'
        f'<span class="{safety_chip_class}">{d.safety_rating} Safety</span>'
        f'<span class="meta-chip">{seasons_text}</span>'
        f'</div>'
        f'<div class="card-snippet">{d.description[:185]}...</div>'
        f'<div class="card-rationale"><b>Why Roamio Chose This:</b> {top_reason}</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    with st.expander(f"Explore {d.name} — Full Overview & Ranking Evidence", expanded=False):
        st.markdown(f"#### {d.name}, {d.country}")
        st.markdown(f"{d.description}")

        if d.cultural_significance:
            st.markdown(f"**Heritage & Cultural Significance**:\n{d.cultural_significance}")

        col1, col2 = st.columns(2)
        with col1:
            if d.activities:
                st.markdown("**Top Activities & Things to Do**:")
                for act in d.activities:
                    st.markdown(f"- {act}")
        with col2:
            if d.famous_foods:
                st.markdown("**Signature Culinary Highlights**:")
                for food in d.famous_foods:
                    st.markdown(f"- {food}")

        st.markdown("---")
        st.markdown("#### Model Decision Reasoning & Grounded Evidence")
        st.markdown(f"- **Trip Cost Estimate**: Approx. ₹{est_trip_cost:,.0f} for {duration_days or 7} days (₹{d.est_daily_cost_inr:,.0f}/day) — *{item.explanation.budget_fit}*")
        st.markdown(f"- **Seasonal Timing**: {', '.join(d.best_seasons)} — *{item.explanation.seasonal_fit}*")
        st.markdown(f"- **Safety Classification**: {d.safety_rating} rating.")

        st.markdown("**Deterministic Feature Signals Considered**:")
        for r in reasons_list:
            st.markdown(f"- {r}")

        if item.explanation.feature_contributions:
            st.markdown("**Relative Signal Contributions**:")
            cols = st.columns(len(item.explanation.feature_contributions))
            for idx, (sig_name, sig_val) in enumerate(item.explanation.feature_contributions.items()):
                with cols[idx]:
                    st.metric(label=sig_name, value=f"{sig_val * 100:.0f}%")

        if show_map_button:
            if st.button(f"Focus {d.name} on Map", key=f"{key_prefix}_focus_{d.destination_id}", use_container_width=True):
                st.session_state["selected_dest_id"] = d.destination_id
                st.rerun()


# Main Tabs: Put Conversational Concierge FRONT AND CENTER (Tab 1)
tab_chat, tab_explore, tab_map, tab_about = st.tabs([
    "Chat Concierge",
    "Explore & Filter",
    "Spatial Map",
    "System Architecture"
])

# ==========================================
# TAB 1: CHAT CONCIERGE (DEFAULT HERO VIEW)
# ==========================================
with tab_chat:
    st.markdown("### Conversational Concierge")
    st.write("Tell Roamio what kind of journey you envision. State your budget, duration, preferred travel month, or activities in everyday language.")

    # Sample Prompts
    st.markdown("**Sample queries to get started:**")
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        if st.button("Budget beach trip in Asia for 8 days", use_container_width=True):
            st.session_state["chat_pending"] = "I have a budget of ₹70,000 for 8 days. I want a warm beach destination in Asia with great food and relaxed vibes."
    with q_col2:
        if st.button("Peaceful mountain retreat with temples", use_container_width=True):
            st.session_state["chat_pending"] = "I am looking for a peaceful mountain getaway with scenic hiking trails, ancient temples, and rich heritage."
    with q_col3:
        if st.button("Historic European cultural city tour", use_container_width=True):
            st.session_state["chat_pending"] = "Recommend historic European cities known for world-class museums, art, architecture, and walkable centers."

    # Chat History
    for msg in chat_session.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input
    user_input = st.chat_input("Where would you like to travel? (e.g. '₹60,000 budget for 7 days in Europe')")
    if "chat_pending" in st.session_state and st.session_state["chat_pending"]:
        user_input = st.session_state.pop("chat_pending")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing preferences & computing hybrid recommendations..."):
                reply, rec_resp = chat_session.process_turn(user_input)
                st.markdown(reply)
                st.session_state["chat_rec_response"] = rec_resp

    # Render Visual Recommendations for Latest Assistant Turn
    chat_rec = st.session_state.get("chat_rec_response")
    if chat_rec and chat_rec.items:
        st.markdown("---")
        st.markdown("#### Curated Recommendations")
        
        c_cards, c_map = st.columns([1.15, 1.0], gap="large")
        with c_cards:
            dur = chat_rec.applied_preferences.duration_days or 7
            for item in chat_rec.items[:4]:
                render_destination_card(item, key_prefix="chat", duration_days=dur, show_map_button=False)

        with c_map:
            st.markdown("<div style='font-size:0.95rem; font-weight:600; color:#172B3A; margin-bottom:0.4rem;'>Geographic Locations</div>", unsafe_allow_html=True)
            chat_map = render_folium_map([it.destination for it in chat_rec.items[:4]], height=440)
            if chat_map:
                st_folium(chat_map, returned_objects=[], use_container_width=True, height=440)

            # Refinement actions
            st.markdown("#### Conversational Refinements")
            ref_col1, ref_col2 = st.columns(2)
            with ref_col1:
                if st.button("Find budget alternatives", use_container_width=True):
                    st.session_state["chat_pending"] = "Actually, show me cheaper budget-friendly options."
                    st.rerun()
                if st.button("Explore other regions", use_container_width=True):
                    st.session_state["chat_pending"] = "Show me alternative destinations in other regions."
                    st.rerun()
            with ref_col2:
                if st.button("Why this top match?", use_container_width=True):
                    st.session_state["chat_pending"] = "Why did you rank the first destination highest?"
                    st.rerun()
                if st.button("Compare top choices", use_container_width=True):
                    st.session_state["chat_pending"] = "Compare the first and second recommendations."
                    st.rerun()


# ==========================================
# TAB 2: EXPLORE & FILTER (SPLIT VIEW)
# ==========================================
with tab_explore:
    with st.expander("Search Filters & Travel Criteria", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            continents_opts = ["All", "Asia", "Europe", "Americas", "Africa", "Middle East", "Oceania"]
            sel_continent = st.selectbox("Continent", continents_opts, key="exp_continent")
        with f2:
            cat_opts = ["All", "City", "Beach", "Mountain", "Cultural", "Archaeological Site", "National Park", "Island", "Lake", "Fjord"]
            sel_category = st.selectbox("Category", cat_opts, key="exp_category")
        with f3:
            budget_val = st.slider("Total Budget (INR)", min_value=15000, max_value=300000, value=80000, step=5000, key="exp_budget")
        with f4:
            duration_val = st.slider("Duration (Days)", min_value=3, max_value=30, value=7, key="exp_duration")

        search_query = st.text_input(
            "Keyword Desires (Optional)",
            placeholder="e.g. serene mountain trails, historic temples, authentic food, coastal relaxation",
            key="exp_query"
        )

        b_col1, b_col2 = st.columns([1, 4])
        with b_col1:
            search_clicked = st.button("Search Destinations", type="primary", use_container_width=True)
        with b_col2:
            if st.button("Reset Criteria", use_container_width=False):
                st.session_state["explore_response"] = None
                st.session_state["selected_dest_id"] = None
                st.rerun()

    if search_clicked or st.session_state["explore_response"] is None:
        user_prefs = UserPreferences(
            query_text=search_query,
            budget_max_inr=float(budget_val),
            duration_days=duration_val,
            continents=[sel_continent] if sel_continent != "All" else [],
            categories=[sel_category] if sel_category != "All" else []
        )
        st.session_state["explore_response"] = engine.recommend(user_prefs, top_k=6, apply_diversity=True)

    rec_res: RecommendationResponse = st.session_state["explore_response"]

    if rec_res and rec_res.items:
        items = rec_res.items
        dest_objs = [it.destination for it in items]

        st.markdown(f"<div style='color:#66737D; font-size:0.9rem; margin-bottom:1rem;'>Showing {len(items)} curated destinations matching your criteria</div>", unsafe_allow_html=True)

        col_cards, col_map = st.columns([1.15, 1.0], gap="large")

        with col_cards:
            dur = duration_val or 7
            for item in items:
                render_destination_card(item, key_prefix="explore", duration_days=dur, show_map_button=True)

        with col_map:
            st.markdown("<div style='font-size:0.95rem; font-weight:600; color:#172B3A; margin-bottom:0.4rem;'>Interactive Destination Map</div>", unsafe_allow_html=True)
            active_id = st.session_state.get("selected_dest_id")
            m = render_folium_map(dest_objs, active_id=active_id, height=560)
            if m:
                st_folium(m, returned_objects=[], use_container_width=True, height=560)

        if st.session_state.get("selected_dest_id"):
            selected_dest = get_destination_by_id(st.session_state["selected_dest_id"])
            if selected_dest:
                st.markdown("---")
                matching_exp = next((it.explanation for it in items if it.destination.destination_id == selected_dest.destination_id), None)
                render_destination_detail_modal(selected_dest, explanation=matching_exp)


# ==========================================
# TAB 3: SPATIAL MAP EXPLORER
# ==========================================
with tab_map:
    st.markdown("### Spatial Map Explorer")
    st.write("Browse destinations across the globe. Click any pin on the map to inspect location descriptions, or filter by continent, category, and budget tier.")

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        map_cont = st.selectbox("Filter Continent", ["All", "Europe", "Asia", "Americas", "Africa", "Middle East", "Oceania"], key="map_filter_cont")
    with m_col2:
        map_cat = st.selectbox("Filter Category", ["All", "City", "Beach", "Mountain", "Cultural", "Archaeological Site", "National Park", "Island"], key="map_filter_cat")
    with m_col3:
        map_cost = st.selectbox("Budget Tier", ["All", "Low", "Medium", "Luxury"], key="map_filter_cost")

    filtered_dests = all_destinations
    if map_cont != "All":
        filtered_dests = [d for d in filtered_dests if d.continent == map_cont]
    if map_cat != "All":
        filtered_dests = [d for d in filtered_dests if map_cat.lower() in d.category.lower()]
    if map_cost != "All":
        filtered_dests = [d for d in filtered_dests if d.cost_level.lower() == map_cost.lower()]

    st.markdown(f"<div style='color:#66737D; font-size:0.88rem; margin-bottom:0.6rem;'>Displaying {len(filtered_dests)} destinations on the map &middot; Click any pin to inspect complete description and photos</div>", unsafe_allow_html=True)

    active_spatial_id = st.session_state.get("spatial_selected_id")
    full_map = render_folium_map(filtered_dests, active_id=active_spatial_id, height=540, zoom_start=2)

    clicked_dest = None
    if full_map:
        map_data = st_folium(
            full_map,
            returned_objects=["last_object_clicked_tooltip", "last_object_clicked"],
            use_container_width=True,
            height=540,
            key="spatial_explorer_map"
        )
        if map_data and map_data.get("last_object_clicked_tooltip"):
            clicked_tooltip = map_data["last_object_clicked_tooltip"]
            for d in all_destinations:
                if f"{d.name}, {d.country}" == clicked_tooltip or d.name == clicked_tooltip or d.name in clicked_tooltip:
                    clicked_dest = d
                    st.session_state["spatial_selected_id"] = d.destination_id
                    break

    # Determine destination to inspect
    active_inspect_dest = clicked_dest
    if not active_inspect_dest and active_spatial_id:
        for d in all_destinations:
            if d.destination_id == active_spatial_id:
                active_inspect_dest = d
                break

    st.markdown("---")
    st.markdown("#### Destination Inspector")

    col_sel, col_stats = st.columns([2.5, 1.5])
    with col_sel:
        dest_picker_options = ["Click a pin above or choose destination..."] + [f"{d.name}, {d.country} ({d.category})" for d in filtered_dests]
        current_picker_idx = 0
        if active_inspect_dest:
            opt_match = f"{active_inspect_dest.name}, {active_inspect_dest.country} ({active_inspect_dest.category})"
            if opt_match in dest_picker_options:
                current_picker_idx = dest_picker_options.index(opt_match)
        
        chosen_opt = st.selectbox("Inspect Destination Details:", dest_picker_options, index=current_picker_idx, key="spatial_dest_picker")
        if chosen_opt != "Click a pin above or choose destination...":
            chosen_dest_name = chosen_opt.split(", ")[0].strip()
            for d in filtered_dests:
                if d.name == chosen_dest_name:
                    active_inspect_dest = d
                    st.session_state["spatial_selected_id"] = d.destination_id
                    break

    with col_stats:
        if active_inspect_dest:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #E4E0D8; border-radius:8px; padding:10px 14px; margin-top:24px;">
                <div style="font-size:12px; color:#66737D;">Daily Budget Profile</div>
                <div style="font-size:16px; font-weight:700; color:#C96F4A;">₹{active_inspect_dest.est_daily_cost_inr:,.0f} <span style="font-size:12px; color:#172B3A; font-weight:400;">/ day</span></div>
                <div style="font-size:11px; color:#55745A; margin-top:2px;">{active_inspect_dest.safety_rating} Safety Prior &middot; {active_inspect_dest.cost_level} Budget</div>
            </div>
            """, unsafe_allow_html=True)

    if active_inspect_dest:
        render_destination_detail_modal(active_inspect_dest, explanation=None)
        
        c_act1, _ = st.columns([1.5, 2])
        with c_act1:
            if st.button(f"Plan a trip to {active_inspect_dest.name} in Concierge Chat", use_container_width=True):
                st.session_state["chat_pending"] = f"I am interested in traveling to {active_inspect_dest.name}, {active_inspect_dest.country}. Can you give me a personalized itinerary, top attractions, and budget breakdown?"
                st.rerun()


# ==========================================
# TAB 4: SYSTEM ARCHITECTURE & EVALUATION
# ==========================================
with tab_about:
    st.markdown("### System Architecture & Empirical Evaluation")
    st.write("Roamio combines conversational intent understanding with two-stage retrieval, semantic embeddings, and multi-signal ranking.")

    st.markdown("""
    #### Architecture Overview:
    1. **Two-Stage Candidate Pipeline**: SQL-based pre-filtering rapidly subsets candidate space, allowing dense semantic scoring and MMR diversity re-ranking to run in sub-15ms latency.
    2. **Dual Representation**: Combines lexical precision (**TF-IDF**) with deep semantic similarity (**BGE / MiniLM Dense Sentence Embeddings**).
    3. **Multi-Signal Hybrid Scorer**: Computes a principled weighted combination of semantic similarity, lexical overlap, budget compatibility decay curves, seasonal alignment, and safety priors.
    4. **Maximal Marginal Relevance (MMR)**: Balances relevance with geographic and categorical diversity to avoid localized recommendation clustering.
    5. **Anti-Hallucination Guardrails**: The LLM never decides destination rankings directly; it translates user intent and renders grounded explanations based on verified catalog features.
    """)

    st.markdown("#### Empirical Benchmark & Ablation Study")
    st.write("Evaluated across curated ground-truth travel benchmarks (Precision@5, Recall@10, NDCG@10, MRR, Intra-List Diversity):")

    if st.button("Run Live Benchmark", use_container_width=False):
        with st.spinner("Executing live ablation benchmark across baseline models..."):
            results = run_ablation_study()
            st.session_state["ablation_results"] = results

    if "ablation_results" in st.session_state:
        df_res = pd.DataFrame(st.session_state["ablation_results"])
        st.dataframe(df_res, use_container_width=True)
    else:
        default_benchmark_data = [
            {"Model": "TF-IDF Baseline", "P@5": 0.2500, "R@10": 0.3438, "NDCG@10": 0.3024, "MRR": 0.4458, "Diversity (ILD)": 0.8562, "Latency (ms)": 1.07},
            {"Model": "Dense Embeddings", "P@5": 0.4750, "R@10": 0.6125, "NDCG@10": 0.6149, "MRR": 0.7333, "Diversity (ILD)": 0.8321, "Latency (ms)": 4.42},
            {"Model": "Two-Stage Retrieval", "P@5": 0.4250, "R@10": 0.6000, "NDCG@10": 0.6169, "MRR": 0.8333, "Diversity (ILD)": 0.7803, "Latency (ms)": 8.72},
            {"Model": "Hybrid (No Diversity)", "P@5": 0.5500, "R@10": 0.7312, "NDCG@10": 0.7585, "MRR": 1.0000, "Diversity (ILD)": 0.8614, "Latency (ms)": 9.86},
            {"Model": "Roamio Hybrid + MMR", "P@5": 0.5000, "R@10": 0.5750, "NDCG@10": 0.6694, "MRR": 1.0000, "Diversity (ILD)": 0.8858, "Latency (ms)": 13.37},
        ]
        st.dataframe(pd.DataFrame(default_benchmark_data), use_container_width=True)

    st.markdown("""
    #### Key Empirical Findings:
    - **Dense Semantic Gain**: Dense embeddings deliver **+103% NDCG@10 improvement** over lexical TF-IDF by recognizing conceptual synonyms (*"serene alpine retreat"* matches *"peaceful mountain"*).
    - **Budget & Seasonal Soft Constraints**: Unconstrained semantic search frequently suggests high-cost luxury getaways to budget backpackers. Roamio's hybrid scorer enforces continuous budget decay curves and seasonal compatibility.
    - **MMR Intra-List Diversity**: Pure similarity tends to cluster all top recommendations within a single region. MMR guarantees regional and categorical variety without sacrificing top match relevance.
    """)

# Minimalist Footer
st.markdown("---")
st.markdown(f"<div style='display:flex; justify-content:space-between; color:#66737D; font-size:0.8rem;'><div>Roamio &bull; Conversational Hybrid Travel Discovery</div><div>Catalog: {db_stats['total_destinations']} destinations across {db_stats['total_countries']} countries</div></div>", unsafe_allow_html=True)
