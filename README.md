# ✈️ Roamio — Conversational Hybrid Travel Recommendation System

[![Roamio CI](https://github.com/mohikarathi/Roamio/actions/workflows/ci.yml/badge.svg)](https://github.com/mohikarathi/Roamio/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![FastEmbed](https://img.shields.io/badge/FastEmbed-ONNX-orange.svg)](https://github.com/qdrant/fastembed)

> **Roamio** is an intelligent, conversational hybrid travel recommendation system that bridges natural language conversational planning with deterministic, multi-signal recommendation algorithms. Evolving from a static single-file prototype into an end-to-end ML application, Roamio demonstrates modern data engineering, lexical and dense retrieval, multi-criteria ranking, diversity optimization, and explainable AI.

---

## 📑 Table of Contents
- [The Problem & Motivation](#-the-problem--motivation)
- [System Architecture](#-system-architecture)
- [Data Pipeline & Multi-Source Provenance](#-data-pipeline--multi-source-provenance)
- [Mathematical Foundations & ML Modeling](#-mathematical-foundations--ml-modeling)
  - [Lexical Retrieval (TF-IDF)](#1-lexical-retrieval-tf-idf)
  - [Dense Semantic Search (Sentence Embeddings)](#2-dense-semantic-search-sentence-embeddings)
  - [Two-Stage Retrieval Pipeline](#3-two-stage-retrieval-pipeline)
  - [Multi-Signal Hybrid Ranking](#4-multi-signal-hybrid-ranking)
  - [Maximal Marginal Relevance (MMR) for Diversity](#5-maximal-marginal-relevance-mmr-for-diversity)
- [Explainability Engine](#-explainability-engine)
- [Conversational Layer & Anti-Hallucination Guardrails](#-conversational-layer--anti-hallucination-guardrails)
- [Empirical Evaluation & Ablation Study](#-empirical-evaluation--ablation-study)
- [Error Analysis & Insights](#-error-analysis--insights)
- [Project Structure](#-project-structure)
- [Installation & Quickstart](#-installation--quickstart)
- [Docker Deployment](#-docker-deployment)
- [The ML Engineering Story (Interview Q&A)](#-the-ml-engineering-story)

---

## 🎯 The Problem & Motivation

Most travel recommendation platforms suffer from one of two extremes:
1. **Rigid Boolean Search Filters**: Require users to manually check dozens of filter dropdowns. Searching for a *"peaceful mountain retreat with temples and great local food under ₹70,000"* requires impossible combinations that frequently result in **0 results** if a single rigid tag misses.
2. **Unconstrained LLM Chatbots**: Freeform chatbots (e.g. vanilla ChatGPT wrappers) sound persuasive but hallucinate fake prices, recommend non-existent itineraries, have no access to catalog inventory, and cannot deterministically rank candidates according to mathematical scoring criteria.

### The Original Roamio Baseline vs. Roamio 2.0
In its initial form, Roamio relied on:
- A single static Excel spreadsheet of 209 destinations across only 21 countries with 24% missing descriptions.
- A multi-class `LogisticRegression` model attempting to classify 208 unique destination names from `(Country, Category)`. **This achieved an empirical accuracy of 0.0%**, as destination recommendation was fundamentally misformulated as multi-class classification rather than **candidate retrieval and ranking**.
- Fragmented, ad-hoc pairwise filter scripts that broke candidate diversity.

**Roamio 2.0 transforms this codebase into a production-style hybrid system:**
- **Multi-source Data Lake & Canonical SQLite DB**: Enriched with curated Wikidata & GeoNames entities across 41 countries with full provenance.
- **Two-Stage Dual Retrieval**: Prunes candidate space with structured SQL filtering, then retrieves candidates using both lexical **TF-IDF** and **Dense Sentence Transformers** (`bge-small-en-v1.5`).
- **Principled Multi-Feature Scorer**: Blends semantic similarity, keyword overlap, continuous budget decay curves, seasonal alignment, and quality priors.
- **Maximal Marginal Relevance (MMR)**: Re-ranks top candidates to eliminate repetitive geographic clusters.
- **Conversational Concierge Layer**: Uses an LLM (Gemini API with an offline fallback parser) strictly as a bidirectional natural language interface. **The LLM never invents recommendations**; it translates user intent into structured constraints and formats deterministic recommendation features into natural conversational responses.

---

## 🏗️ System Architecture

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
                                   │  Extracts Structured Intent
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
│  Filtering   │          │ (Lexical Match) │         │ (bge-small-en)  │
└──────┬───────┘          └────────┬────────┘         └────────┬────────┘
       │                           │                           │
       └───────────────────────────┼───────────────────────────┘
                                   ▼
                       ┌───────────────────────┐
                       │    Hybrid Ranking     │
                       │ w1*Dense + w2*Lexical │
                       │ + w3*Pref + w4*Budget │
                       │ + w5*Season + w6*Qual │
                       └───────────┬───────────┘
                                   ▼
                       ┌───────────────────────┐
                       │ Maximal Marginal Rel. │
                       │ (Diversity Re-rank)   │
                       └───────────┬───────────┘
                                   ▼
                       ┌───────────────────────┐
                       │ Explainability Engine │
                       │ (Attribution Vectors) │
                       └───────────┬───────────┘
                                   ▼
                       ┌───────────────────────┐
                       │ Top-K + Reasons + LLM │
                       │  Grounded Response    │
                       └───────────────────────┘
```

---

## 🗄️ Data Pipeline & Multi-Source Provenance

Roamio moves away from reading static Excel spreadsheets at runtime. A reproducible data engineering pipeline (`python -m src.data.pipeline`) automates ingestion, deduplication, validation, and precomputation:

```
  Existing destinations.xlsx (Seed)
               +
  Wikidata / Wikipedia API (Curated Global Destinations)
               +
  GeoNames / OpenStreetMap Metadata (Coords, Climate, Country)
               │
               ▼
      data/raw/ (JSON/CSV with Provenance Metadata)
               │
               ▼
      src/data/cleaner.py (Normalize & Sanitize)
               │
               ▼
      src/data/entity_resolution.py (Canonical IDs, Fuzzy Match)
               │
               ▼
      src/data/validator.py (Pydantic Schema Validation)
               │
               ▼
      SQLite Database: data/roamio.db
      (destinations, categories, climate, sources)
               │
               ▼
      src/retrieval/corpus_builder.py
      (Rich semantic document representation per destination)
               │
       ┌───────┴───────┐
       ▼               ▼
  Precomputed     Precomputed
  TF-IDF Matrix   Dense Embeddings (Cache: data/processed/)
```

### Data Sources & Provenance Matrix

| Source | Identifier / Type | Scope | Usage & License | Update Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline Seed** | `attached_assets/destinations.xlsx` | 209 destinations across 21 countries | Historic seed data; preserved for baseline benchmarking | Static baseline |
| **Wikidata / OpenTravel** | Wikidata QIDs (e.g. `Q34647` Kyoto) | ~30 world icons across Asia, Europe, Americas, Africa | Curated coordinates, UNESCO tags, cultural heritage | Ingested via pipeline |
| **GeoNames / Open-Meteo** | Spatial Coordinates & Monthly Climate | Global climate envelopes & coordinates | CC-BY / Open Access | Cached in SQLite |

### Entity Resolution
Different sources reference destinations with slight orthographic or administrative variations (`Kyoto`, `Kyoto City`, `Kyoto, Japan`). The `EntityResolver` performs:
1. **Authoritative External ID Match**: Exact matches on Wikidata QIDs.
2. **Normalized Name & Country Match**: Strips noise words (`City`, `Province`, `Prefecture`, diacritics).
3. **Spatial Proximity & Token Set Similarity**: If Haversine distance $< 35\text{ km}$ and token Jaccard similarity $\ge 0.65$, records are merged into a canonical entity, uniting activity tags and preserving the richer description.

---

## 📐 Mathematical Foundations & ML Modeling

### 1. Lexical Retrieval (TF-IDF)
The lexical retriever indexes rich multi-field semantic documents encompassing destination name, country, category, tags, narrative overview, activities, famous foods, and seasons:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \left( \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1 \right)$$

Cosine similarity between the query $\vec{q}$ and candidate document $\vec{d}$:

$$\text{Sim}_{\text{lexical}}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\|_2 \|\vec{d}\|_2}$$

### 2. Dense Semantic Search (Sentence Embeddings)
To resolve vocabulary mismatch (e.g. query *"serene alpine retreat"* vs document *"peaceful mountain resort"*), Roamio precomputes 384-dimensional dense vectors using `BAAI/bge-small-en-v1.5` via ONNX Runtime (`fastembed`):

$$\vec{e}_d = \frac{\text{Transformer}(d)}{\|\text{Transformer}(d)\|_2}, \quad \vec{e}_q = \frac{\text{Transformer}(q)}{\|\text{Transformer}(q)\|_2}$$

Because vectors are unit-normalized, cosine similarity reduces to a fast dot product:

$$\text{Sim}_{\text{dense}}(\vec{q}, \vec{d}) = \vec{e}_q \cdot \vec{e}_d$$

### 3. Two-Stage Retrieval Pipeline
Rather than performing unconstrained semantic search over the entire universe:
- **Stage 1 (Structured Candidate Filtering)**: Fast SQL queries prune candidate space based on hard constraints (e.g. continent, safety rating, travel month). A soft-relaxation fallback guarantees recall never drops to zero.
- **Stage 2 (Dual Scoring)**: Evaluates both $\text{Sim}_{\text{dense}}$ and $\text{Sim}_{\text{lexical}}$ over the filtered candidate pool in under $10\text{ ms}$.

### 4. Multi-Signal Hybrid Ranking
A destination cannot be recommended solely on text similarity; budget and seasonality are critical feasibility constraints. The `HybridScorer` computes normalized signals $[0.0, 1.0]$:

$$\text{Final Score}(d) = w_{\text{dense}} S_{\text{dense}} + w_{\text{tfidf}} S_{\text{tfidf}} + w_{\text{pref}} S_{\text{pref}} + w_{\text{budget}} S_{\text{budget}} + w_{\text{season}} S_{\text{season}} + w_{\text{qual}} S_{\text{qual}}$$

#### Continuous Budget Decay Curve
Given user trip budget $B$ and duration $D$, the daily budget target is $d = 0.80 \cdot B / D$. If the destination estimated daily cost is $c$:
- If $c \le d$: $S_{\text{budget}} = 1.0 - 0.15 \cdot \frac{d - c}{d}$
- If $c > d$: $S_{\text{budget}} = \exp\left(-2.5 \cdot \frac{c - d}{d}\right)$

This ensures destinations slightly above budget receive a modest penalty, while destinations costing $2\times$ the target decay toward zero.

### 5. Maximal Marginal Relevance (MMR) for Diversity
Recommending 5 destinations all located within the same province creates a poor user experience. Roamio balances relevance with diversity using MMR:

$$\text{Next} = \arg\max_{d_i \in R \setminus S} \left[ \lambda \cdot \text{Score}(d_i) - (1 - \lambda) \cdot \max_{d_j \in S} \text{Sim}_{\text{inter}}(d_i, d_j) \right]$$

where $\text{Sim}_{\text{inter}}$ measures pairwise geographic Haversine distance, country collision, and category overlap. Setting $\lambda = 0.75$ provides an optimal balance between topical precision and regional variety.

---

## 🔍 Explainability Engine

Every recommendation returns verified, evidence-based rationales grounded strictly in underlying feature vectors:

```json
{
  "rank": 1,
  "destination": "Ella, Sri Lanka",
  "score": 0.6584,
  "reasons": [
    "Strong semantic alignment with your preferred travel experience",
    "Direct match for your interests: hiking, peaceful",
    "Comfortably within budget (Est. ₹2,400/day vs ₹7,000/day target)"
  ],
  "feature_contributions": {
    "Semantic Match": 0.616,
    "Keyword Match": 0.071,
    "Preferences": 0.750,
    "Budget Fit": 0.900,
    "Season Fit": 0.700,
    "Quality/Safety": 0.820
  }
}
```
In the Streamlit interface, users can expand any card to view the exact percentage contribution of each signal.

---

## 💬 Conversational Layer & Anti-Hallucination Guardrails

The conversational interface supports natural, multi-turn trip planning:
- **Intent Extraction**: Translates freeform queries into validated `UserPreferences` objects (`budget_max_inr`, `duration_days`, `travel_month`, `interests`, `dislikes`).
- **Conversational Refinement**:
  - *"Actually, make it cheaper"* $\rightarrow$ reduces budget threshold and triggers reranking.
  - *"Why did you rank Rio de Janeiro at #1?"* $\rightarrow$ inspects feature attribution vectors to explain the ranking decision.
  - *"Compare 1 and 2"* $\rightarrow$ formats a side-by-side comparative feature matrix.
- **Strict Anti-Hallucination Guarantee**: The LLM operates **around** the recommendation engine, not within it. It never selects candidate rankings or invents destinations. If external API keys are unavailable, Roamio seamlessly falls back to a deterministic rule-based parser with zero degradation of recommendation quality.

---

## 📊 Empirical Evaluation & Ablation Study

Roamio features an offline evaluation harness (`src/evaluation/`) benchmarking models across 8 realistic travel personas with curated ground truth.

### Benchmark Ablation Results

| Model / Configuration | Precision@5 | Recall@10 | NDCG@10 | MRR | Diversity (ILD) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Baseline** | 0.2500 | 0.3438 | 0.3024 | 0.4458 | 0.8562 | **1.07 ms** |
| **Dense Embeddings** | 0.4750 | 0.6125 | 0.6149 | 0.7333 | 0.8321 | 4.42 ms |
| **Two-Stage Retrieval** | 0.4250 | 0.6000 | 0.6169 | 0.8333 | 0.7803 | 8.72 ms |
| **Hybrid (No Diversity)** | **0.5500** | **0.7312** | **0.7585** | **1.0000** | 0.8614 | 9.86 ms |
| **Roamio Hybrid + MMR** | 0.5000 | 0.5750 | 0.6694 | **1.0000** | **0.8858** | 13.37 ms |

### Key Findings:
1. **Dense vs. Lexical (+103% NDCG@10 gain)**: Dense embeddings capture conceptual synonyms (*"alpine retreat"* matches *"mountain resort"*), whereas TF-IDF misses unless exact tokens overlap.
2. **Two-Stage Filtering (+13% MRR gain)**: Pruning candidates by continent and safety eliminates geographically invalid results.
3. **Hybrid Scoring Engine (MRR = 1.0000)**: Integrating continuous budget decay curves and seasonal alignment guarantees that a highly relevant destination appears at Rank 1 for every benchmark query.
4. **MMR Diversity (+3% ILD gain)**: Enforces catalog variety, preventing recommendations from collapsing into a single geographic region.

---

## 🔬 Error Analysis & Insights

- **Where TF-IDF Outperforms Embeddings**: Queries featuring rare proper nouns (e.g. *"Kinkaku-ji"* or *"Matterhorn"*) benefit from TF-IDF's exact token matching, where embeddings can sometimes suffer from semantic drift.
- **Where Pure Embeddings Fail**: Dense embeddings are budget-blind. A user requesting a *"luxury beachfront overwater villa"* will retrieve the Maldives even if their budget is ₹20,000. Hybrid scoring fixes this by penalizing budget-incompatible candidates.
- **The Diversity vs. Precision Trade-Off**: Increasing MMR diversity ($\lambda \rightarrow 0.5$) slightly reduces top-5 precision in exchange for broader regional exploration.

---

## 📂 Project Structure

```
Roamio/
├── data/
│   ├── raw/                       # Immutable raw JSON/CSV data with provenance
│   ├── processed/                 # Cached TF-IDF & dense embeddings (.npy, .pkl)
│   ├── metadata/                  # Data validation reports & audit logs
│   └── roamio.db                  # Canonical SQLite database
├── src/
│   ├── config.py                  # Global paths, weights, and constants
│   ├── data/
│   │   ├── models.py              # Pydantic schemas (Destination, UserPreferences)
│   │   ├── normalizer.py          # Temporal, budget, and string sanitizers
│   │   ├── entity_resolution.py   # Multi-source deduplication & canonical IDs
│   │   ├── validator.py           # Data validation schema checks
│   │   ├── ingestion.py           # Multi-source dataset ingestion
│   │   ├── pipeline.py            # Reproducible data & model build pipeline
│   │   └── images/                # Automated landscape imagery acquisition
│   │       ├── curated.py         # Curated high-res photo registry with attribution
│   │       └── pipeline.py        # CLI module for Unsplash/Pexels + fallback caching
│   ├── retrieval/
│   │   ├── base.py                # Abstract retriever interface
│   │   ├── tfidf.py               # TF-IDF lexical baseline retriever
│   │   ├── dense.py               # FastEmbed dense semantic search
│   │   └── two_stage.py           # Candidate filtering + dual retrieval
│   ├── ranking/
│   │   ├── scorer.py              # Multi-feature hybrid ranking engine
│   │   ├── diversity.py           # Maximal Marginal Relevance (MMR)
│   │   ├── explain.py             # Feature attribution explainability engine
│   │   └── engine.py              # Master Recommendation API service
│   ├── chat/
│   │   ├── client.py              # LLM provider abstraction (Gemini + Offline fallback)
│   │   └── session.py             # Multi-turn state, refinement, and comparisons
│   └── evaluation/
│       ├── metrics.py             # Precision@K, Recall@K, NDCG@K, MRR, ILD
│       ├── benchmark.py           # Curated travel persona query suite
│       └── runner.py              # Automated ablation runner
├── notebooks/                     # 5 in-depth walkthrough experiment notebooks
├── tests/                         # Full Pytest test suite (100% pass rate)
├── app.py                         # Modernized Streamlit Web Application
├── Dockerfile                     # Multi-stage production container
├── docker-compose.yml             # Local Docker Compose setup
├── requirements.txt               # Pinned production dependencies
└── .github/workflows/ci.yml       # Automated GitHub Actions CI workflow
```

---

## 🚀 Installation & Quickstart

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/mohikarathi/Roamio.git
cd Roamio

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Data Pipeline & Precompute Embeddings
```bash
python -m src.data.pipeline
```
*(Runs ingestion, entity resolution, validation, SQLite population, and precomputes embeddings in ~20 seconds).*

### 3. Fetch Landscape Photography (100% Catalog Coverage)
```bash
python -m src.data.images.pipeline --missing-only
```
*(Acquires high-resolution landscape travel photography with photographer attribution for all 232 destinations via Unsplash/Pexels or the curated verified registry).*

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Optional: Enable Google Gemini API
To use Gemini for natural language response formatting, set your API key:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```
*Note: If no API key is provided, Roamio automatically runs using its high-speed offline rule-based parser with zero setup.*

---

## 🐳 Docker Deployment

Roamio includes a production-ready `Dockerfile` and `docker-compose.yml`:

```bash
# Build and start container
docker-compose up --build -d

# View logs
docker-compose logs -f

# Access application at http://localhost:8501
```

---

## 🧪 Running Automated Tests

Run the comprehensive pytest suite:
```bash
PYTHONPATH=. pytest tests/ -v
```

Run the live ablation benchmark:
```bash
PYTHONPATH=. python -m src.evaluation.runner
```

---

## 🎓 The ML Engineering Story

### Why TF-IDF?
TF-IDF acts as our deterministic lexical baseline. For specific proper nouns and exact activity queries (e.g. *"scuba diving in Great Barrier Reef"*), lexical matching is precise, fast ($1.07\text{ ms}$), and computationally lightweight.

### Why Dense Sentence Embeddings?
Lexical matching fails when users express subjective desires (*"peaceful mountain getaway"*). Sentence transformers map queries and destination descriptions into a continuous vector space where semantically synonymous phrases cluster closely together, doubling retrieval NDCG.

### Why Precompute Embeddings?
Destination descriptions are static or slowly changing, whereas user queries arrive dynamically in real time. Precomputing and normalizing destination embeddings once at build time allows query inference to run via a fast vector dot product in $< 5\text{ ms}$, eliminating redundant inference costs.

### Why Hybrid Multi-Criteria Ranking?
Semantic similarity alone cannot verify whether a hotel fits a user's wallet or whether visiting in December will coincide with monsoon season. The hybrid scoring engine combines semantic match with financial decay curves and seasonal alignment.

### Why Not Let the LLM Recommend Directly?
Allowing an LLM to generate recommendations independently introduces severe hallucination risks: invented prices, non-existent destinations, and non-deterministic rankings. Roamio uses the LLM strictly as an **interface**: extracting structured user intent and communicating deterministic recommendation facts.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).