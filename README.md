# Roamio — Curated Travel Guide and Recommendation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://roamiotravelrecommendation.streamlit.app/)
[![Roamio CI](https://github.com/mohikarathi/Roamio/actions/workflows/ci.yml/badge.svg)](https://github.com/mohikarathi/Roamio/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Live Application**: Try Roamio live on Streamlit Cloud at **[roamiotravelrecommendation.streamlit.app](https://roamiotravelrecommendation.streamlit.app/)**.

> **Roamio** is a curated travel guide and intelligent recommendation platform that combines conversational AI planning with multi-signal data retrieval and ranking algorithms. Roamio helps travelers discover destinations through natural language conversation, dynamic preference filtering, interactive maps, and transparent match explanations.

---

## Table of Contents
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Data Retrieval & Candidate Generation](#data-retrieval--candidate-generation)
- [Hybrid Recommendation Engine](#hybrid-recommendation-engine)
- [Conversational Concierge](#conversational-concierge)
- [Data Sources & Destination Catalog](#data-sources--destination-catalog)
- [Evaluation & Benchmark Results](#evaluation--benchmark-results)
- [Project Structure](#project-structure)
- [Installation & Quickstart](#installation--quickstart)
- [Live Deployment](#live-deployment)
- [Automated Tests](#automated-tests)
- [Design Decisions](#design-decisions)
- [License](#license)

---

## Key Features

### 1. Conversational AI Concierge
- **Natural Language Trip Planning**: Chat naturally about desired travel experiences, activities, budgets, and dates (e.g., *"Find a relaxing beach escape in Southeast Asia under 60,000 INR for 7 days"*).
- **Preference Extraction**: Automatically extracts budget limits, trip durations, target months, continent preferences, and activity interests from user messages.
- **Conversational Refinement**: Refine search parameters turn-by-turn (e.g., *"Make it cheaper"*, *"Show me cultural options instead"*).
- **Side-by-Side Comparisons**: Ask to compare destinations (e.g., *"Compare 1 and 2"*) to view a comparative table contrasting daily costs, safety ratings, best seasons, and match scores.
- **Explainable Ranking ("Why #1?")**: Ask *"Why did you rank this first?"* to receive an itemized breakdown of why a destination matched your query.
- **Zero-Network Fallback**: Features an offline rule-based parser that provides recommendation extraction and formatting even if external LLM APIs are unavailable.

### 2. Destination Explorer & Search
- **Multi-Parameter Filtering**: Filter destinations by freeform search queries, total budget, trip duration, continents, and travel categories.
- **Real-Time Recommendation Scoring**: Evaluates candidate destinations against constraints and ranks them by relevance.
- **Detailed Modal Inspection**: Click any destination card to open a full overview displaying cultural significance, signature local foods, top activities, estimated daily costs, and climate suitability.

### 3. Editorial Multi-Photo Galleries
- **Landscape Photography Strip**: Every destination card features a curated four-photo landscape gallery strip showcasing landmarks, beaches, nature, and cultural sites.
- **Direct CDN Delivery**: High-resolution imagery delivered directly via public CDNs (Wikimedia Commons and Unsplash) for fast, reliable browser loading.
- **Photographer Attribution**: Transparent attribution links to source photographers and platforms.

### 4. Interactive Spatial Map Explorer
- **Global Leaflet / Folium Map**: Interactive world map displaying destination markers across continents.
- **Synchronized Map Pins**: Pins display destination names, photos, daily costs, and seasons on click or hover.
- **Faceted Map Filtering**: Filter map pins dynamically by continent, category, and budget tier.

### 5. Two-Stage Data Retrieval
- **Structured Candidate Filtering (Stage 1)**: Rapid SQL-based candidate generation pruning the catalog by continent, season, budget, and safety criteria with automatic fallback relaxation to prevent zero-result outcomes.
- **Dual Lexical & Dense Retrieval (Stage 2)**: Pairs fast TF-IDF keyword indexing for exact entity names and regional dishes with 384-dimensional dense semantic embeddings (`BAAI/bge-small-en-v1.5` via FastEmbed) to capture subjective travel desires.
- **Precomputed Sub-10ms Ingestion**: Offline embedding normalization allows real-time dot-product candidate retrieval in under 10 milliseconds.

### 6. Multi-Signal Hybrid Ranking
- **Semantic Understanding**: Uses dense sentence embeddings to match conceptual travel themes (e.g., *"quiet coastal escape"* matches *"peaceful seaside village"*).
- **Keyword Precision**: Lexical TF-IDF matching captures specific landmark names, regions, and cuisine.
- **Financial Feasibility**: Smooth budget decay curves penalize destinations exceeding the user's budget while rewarding cost-efficient options.
- **Seasonal Compatibility**: Prioritizes destinations during their optimal travel months and climate conditions.
- **Quality and Safety Priors**: Factors in verified safety ratings and destination popularity metrics.

### 7. Diversity Re-ranking (MMR)
- **Maximal Marginal Relevance**: Balances relevance score with intra-list diversity to prevent geographic clustering (e.g., preventing multiple recommendations from the same province).
- **Diverse Discovery**: Ensures travelers explore varied options across different regions and categories.

### 8. Transparent Explainability
- **Grounded Match Reasons**: Explanations highlight why each destination matches specific budget targets, seasonal timing, and activity desires.
- **Feature Contribution Breakdown**: Clear percentage breakdowns show the relative contribution of semantic match, keywords, budget, seasonality, and safety.

---

## System Architecture

```
                                  USER
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │   Streamlit Web UI    │
                       │ (Chat + Filters + Map)│
                       └───────────┬───────────┘
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │  Conversational Layer │
                       │ (Gemini / Rule Parser)│
                       └───────────┬───────────┘
                                   │  Extracts Structured Preferences
                                   ▼  (budget, duration, season, interests)
                       ┌───────────────────────┐
                       │  Recommendation API   │
                       └───────────┬───────────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       ▼                           ▼                           ▼
┌──────────────┐          ┌─────────────────┐         ┌─────────────────┐
│  Structured  │          │ TF-IDF Lexical  │         │ Dense Semantic  │
│  Candidate   │          │    Retrieval    │         │   Embeddings    │
│  Filtering   │          │ (Keyword Match) │         │ (bge-small-en)  │
└──────┬───────┘          └────────┬────────┘         └────────┬────────┘
       │                           │                           │
       └───────────────────────────┼───────────────────────────┘
                                   ▼
                       ┌───────────────────────┐
                       │    Hybrid Ranking     │
                       │ Dense + Lexical Match │
                       │ + Budget + Season Fit │
                       │ + Safety + Popularity │
                       └───────────┬───────────┘
                                   ▼
                       ┌───────────────────────┐
                       │ Maximal Marginal Rel. │
                       │ (Diversity Re-rank)   │
                       └───────────┬───────────┘
                                   ▼
                       ┌───────────────────────┐
                       │ Explainability Engine │
                       │ (Feature Attribution) │
                       └───────────┬───────────┘
                                   ▼
                       ┌───────────────────────┐
                       │ Top-K + Explanations  │
                       │  Grounded Response    │
                       └───────────────────────┘
```

---

## Data Retrieval & Candidate Generation

Roamio utilizes a two-stage data retrieval architecture designed for high candidate recall and low search latency:

### Stage 1: Structured Candidate Filtering
Rather than performing unconstrained vector searches across the entire database, the retrieval pipeline first applies structured SQL filters to enforce hard feasibility constraints:
- **Geographic Filtering**: Narrows the search space to target continents (e.g., Europe, Asia) or specific countries when indicated.
- **Safety Filtering**: Filters out destinations below the user's minimum safety requirement.
- **Seasonal Alignment**: Prioritizes destinations where the selected travel month falls within peak or optimal visiting seasons.
- **Budget Thresholding**: Computes an estimated daily budget from total trip funds and duration, filtering out destinations that exceed the budget ceiling.
- **Soft Relaxation Fallback**: If strict constraints reduce the candidate pool below 15 destinations, the filter dynamically relaxes constraints (e.g., widening budget thresholds or neighboring seasons) to ensure the user is never left with zero results.

### Stage 2: Dual Candidate Retrieval
Surviving candidates are concurrently evaluated by two complementary retrieval models:
1. **Lexical Retrieval (TF-IDF)**:
   - Evaluates token frequencies across a rich semantic document constructed for each destination (name, country, category, tags, overview narrative, attractions, famous foods, and seasons).
   - Excels at exact keyword matching for specific landmark names (e.g., *"Colosseum"*, *"Angkor Wat"*) and regional cuisines.
2. **Dense Semantic Retrieval (Vector Embeddings)**:
   - Uses `BAAI/bge-small-en-v1.5` via ONNX Runtime (`fastembed`) to represent destinations and queries in a 384-dimensional continuous vector space.
   - Resolves vocabulary mismatch by capturing conceptual synonyms (e.g., connecting a query for *"serene mountain retreat"* with a destination described as a *"peaceful alpine sanctuary"*).
   - Uses precomputed, unit-normalized vectors so runtime retrieval reduces to a high-speed matrix dot product taking under 5 milliseconds.

Both retrieval scores are normalized and passed forward to the hybrid ranking engine.

---

## Hybrid Recommendation Engine

Once candidate records are retrieved, Roamio executes a multi-signal scoring and ranking pipeline:

1. **Multi-Criteria Hybrid Scoring**:
   - **Semantic Score**: Evaluates narrative alignment between the user's query and the destination's experiential profile.
   - **Lexical Score**: Measures keyword overlap for specific activities, foods, and cultural attractions.
   - **Budget Fit**: Evaluates destination daily cost against the user's daily budget target, applying continuous decay penalties for higher-cost destinations.
   - **Seasonal Fit**: Rewards destinations during their optimal travel months and applies off-peak adjustments.
   - **Quality and Safety**: Incorporates baseline safety tiers and global popularity indicators.

2. **Intra-List Diversity Optimization (MMR)**:
   - Uses Maximal Marginal Relevance (MMR) to balance individual candidate relevance against pairwise geographic distance and category similarity.
   - Guarantees recommendations offer geographic variety across countries and landscape types, preventing repetitive destination clusters.

3. **Explainability Generation**:
   - Automatically constructs grounded natural-language rationales and percentage contribution weights for each candidate, showing exactly why each destination matched.

---

## Conversational Concierge

The conversational interface acts as an intelligent layer above the recommendation engine:

- **Strict Anti-Hallucination Design**: The language model never invents destinations, prices, or rankings. The recommendation engine deterministically selects and scores all candidates from the catalog; the LLM formats the results and handles conversational dialogue.
- **Multi-Turn State Tracking**: The `ChatSession` maintains conversation history and updates cumulative user preferences across turns.
- **Comparative Analysis**: Parses comparison queries to render structured comparative matrices directly in chat.
- **Offline Reliability**: When API keys are not configured, Roamio defaults to a deterministic intent-extraction parser and template formatter, ensuring complete functionality without internet-dependent LLM services.

---

## Data Sources & Destination Catalog

Roamio combines structured travel data, open geographical databases, climatological records, and open-license photography into a unified, canonical SQLite database (`data/roamio.db`).

### Where the Data Comes From

| Data Source | Type & Coverage | Information Provided | Usage & License |
| :--- | :--- | :--- | :--- |
| **Curated Travel Inventory** | 250+ global destinations across 41 countries | Destination names, regions, primary categories, approximate tourist volumes, and comprehensive descriptions | Open travel datasets |
| **Wikidata & Wikipedia** | Structured knowledge graph (Wikidata QIDs) | UNESCO World Heritage designations, cultural significance, historical landmarks, signature local foods, and top activities | Creative Commons CC0 / CC BY-SA |
| **GeoNames & OpenStreetMap** | Spatial coordinates and administrative hierarchy | Canonical latitude, longitude, continent mappings, country codes, and regional administrative boundaries | CC BY 4.0 / ODbL |
| **Open-Meteo & Climate Data** | Climatological temperature and precipitation records | Monthly weather patterns mapped to optimal and peak visiting seasons (best travel months) | Open Access |
| **Wikimedia Commons** | Open media repository | High-resolution landscape photography of cultural landmarks, architectural sites, and natural wonders with photographer attributions | Creative Commons / Public Domain |
| **Unsplash** | Editorial travel photography | Curated landscape photography covering beaches, mountain ranges, cities, and heritage sites | Unsplash License (free commercial & non-commercial use) |

### Data Pipeline & Normalization
A reproducible data pipeline (`python -m src.data.pipeline`) ingests and unifies these diverse data sources:
1. **Sanitization & Normalization**: Standardizes country and continent names, normalizes descriptions, and parses tourist volume metrics.
2. **Entity Resolution**: Reconciles spelling variations across sources (e.g. matching Wikidata QIDs, or comparing geographic coordinates and token similarity) to prevent duplicate entries.
3. **Financial Estimation**: Estimates standardized daily travel costs per person in INR based on destination budget tier and local purchasing indices.
4. **Validation**: Enforces strict schema integrity via Pydantic (`Destination` model) before populating the database and precomputing search indices.

---

## Evaluation & Benchmark Results

Roamio includes an automated evaluation suite (`src/evaluation/`) benchmarking retrieval and ranking performance across diverse travel personas with curated ground truth.

| Configuration | Precision@5 | Recall@10 | NDCG@10 | MRR | Diversity (ILD) | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| TF-IDF Baseline | 0.2500 | 0.3438 | 0.3024 | 0.4458 | 0.8562 | ~1.1 ms |
| Dense Embeddings | 0.4750 | 0.6125 | 0.6149 | 0.7333 | 0.8321 | ~4.4 ms |
| Two-Stage Retrieval | 0.4250 | 0.6000 | 0.6169 | 0.8333 | 0.7803 | ~8.7 ms |
| Hybrid (No Diversity) | 0.5500 | 0.7312 | 0.7585 | 1.0000 | 0.8614 | ~9.9 ms |
| Roamio Hybrid + MMR | 0.5000 | 0.5750 | 0.6694 | 1.0000 | 0.8858 | ~13.4 ms |

### Key Takeaways:
- **Dense vs. Lexical Search**: Dense embeddings improve semantic retrieval by matching conceptual synonyms (*"peaceful getaway"* matching *"serene retreat"*).
- **Hybrid Scoring**: Blending text similarity with budget decay and seasonal fit achieves an MRR of 1.0000 across benchmark personas, ensuring a viable destination always ranks at #1.
- **MMR Diversity**: Increases intra-list diversity, ensuring recommendations span multiple regions and travel styles.

---

## Project Structure

```
Roamio/
├── data/
│   ├── raw/                       # Immutable raw destination datasets with provenance
│   ├── processed/                 # Cached TF-IDF & dense embeddings (.npy, .pkl)
│   ├── metadata/                  # Data validation reports & audit logs
│   ├── destination_galleries.json # Curated multi-image gallery catalog
│   └── roamio.db                  # Canonical SQLite database
├── src/
│   ├── config.py                  # Global configuration, weights, and constants
│   ├── data/
│   │   ├── models.py              # Pydantic schemas (Destination, UserPreferences)
│   │   ├── normalizer.py          # Temporal, budget, and text sanitizers
│   │   ├── entity_resolution.py   # Deduplication & canonical entity resolution
│   │   ├── validator.py           # Pydantic schema validation checks
│   │   ├── ingestion.py           # Multi-source dataset ingestion
│   │   ├── pipeline.py            # Reproducible data and embedding pipeline
│   │   └── images/                # Landscape photo registries and caching
│   ├── retrieval/
│   │   ├── base.py                # Abstract retriever interface
│   │   ├── tfidf.py               # Lexical TF-IDF retriever
│   │   ├── dense.py               # FastEmbed dense semantic search
│   │   └── two_stage.py           # Two-stage candidate filtering
│   ├── ranking/
│   │   ├── scorer.py              # Multi-feature hybrid ranking engine
│   │   ├── diversity.py           # Maximal Marginal Relevance (MMR)
│   │   ├── explain.py             # Feature attribution explainability engine
│   │   └── engine.py              # Master recommendation API service
│   ├── chat/
│   │   ├── client.py              # LLM client (Gemini with offline fallback)
│   │   └── session.py             # Multi-turn conversation state manager
│   └── evaluation/
│       ├── metrics.py             # Precision@K, Recall@K, NDCG@K, MRR, ILD
│       ├── benchmark.py           # Curated travel persona query suite
│       └── runner.py              # Automated ablation runner
├── notebooks/                     # Exploratory analysis and benchmark notebooks
├── tests/                         # Comprehensive Pytest test suite (36 tests)
├── app.py                         # Streamlit web application
├── Dockerfile                     # Container definition
├── docker-compose.yml             # Docker Compose orchestration
├── requirements.txt               # Pinned Python dependencies
└── .github/workflows/ci.yml       # GitHub Actions CI workflow
```

---

## Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/mohikarathi/Roamio.git
cd Roamio
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Build the Database & Precompute Embeddings
```bash
python -m src.data.pipeline
```
This populates the SQLite database, validates destination schemas, and precomputes the TF-IDF matrix and dense embedding vectors.

### 4. Launch the Streamlit Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. Optional: Configure Gemini API Key
To enable Gemini for natural language response formatting, set your environment variable:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```
If no key is provided, Roamio runs automatically using its built-in rule-based conversational parser.

---

## Live Deployment

Roamio is hosted and continuously deployed on **Streamlit Community Cloud**:

- **Live URL**: [roamiotravelrecommendation.streamlit.app](https://roamiotravelrecommendation.streamlit.app/)
- **Repository Source**: Directly connected to the `main` branch of this GitHub repository with automated redeployment on push.
- **Continuous Integration**: GitHub Actions CI workflow runs automated tests and validation across pull requests.

*(Optional)* A `Dockerfile` and `docker-compose.yml` are also provided in the repository for developers who wish to containerize and run the application locally or self-host on custom cloud infrastructure.

---

## Automated Tests

Run the complete automated test suite:
```bash
PYTHONPATH=. pytest tests/ -v
```

Run the benchmark evaluation runner:
```bash
PYTHONPATH=. python -m src.evaluation.runner
```

---

## Design Decisions

- **Why Dual Retrieval?** Lexical TF-IDF search handles exact landmark names and specific activities, while dense sentence embeddings capture broad experiential concepts (*"serene retreat"* matching *"peaceful temple"*). Combining both ensures high recall across query styles.
- **Why Precomputed Embeddings?** Destination descriptions are precomputed and normalized once at build time. User query vectors are projected in real time and evaluated via dot products in under 5 ms, keeping latency low.
- **Why Multi-Feature Hybrid Scoring?** Recommending travel destinations requires balancing subjective interest with real-world constraints like budget, travel dates, and safety. A hybrid scorer guarantees practical feasibility.
- **Why Separate the LLM from Ranking?** Allowing language models to generate recommendations directly often leads to hallucinated prices, invalid locations, and ungrounded suggestions. In Roamio, the recommendation engine handles candidate scoring, while the LLM acts purely as a conversational interface.
