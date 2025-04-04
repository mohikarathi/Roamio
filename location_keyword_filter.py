
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yake

def train_location_keyword_model(df):
    """Train and prepare the location-keyword based recommendation model."""
    try:
        # Ensure necessary columns exist
        if "latitude" not in df.columns or "longitude" not in df.columns:
            return None, "Dataset must contain latitude and longitude information."

        # Convert lists to strings in the Cultural_Description column
        df["cultural_description"] = df["cultural_description"].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))

        # TF-IDF Vectorizer for cultural descriptions
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(df["cultural_description"].fillna(""))

        # Initialize keyword extractor
        kw_extractor = yake.KeywordExtractor(
            lan="en", 
            n=2,
            dedupLim=0.7,
            top=20
        )

        # Initialize NearestNeighbors for location-based recommendations
        location_model = NearestNeighbors(n_neighbors=min(20, len(df)), metric='haversine', algorithm='auto')
        coords = np.radians(df[['latitude', 'longitude']].values)
        location_model.fit(coords)

        return {
            'vectorizer': vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'kw_extractor': kw_extractor,
            'location_model': location_model,
            'data': df
        }, None
    except Exception as e:
        return None, f"Error training model: {str(e)}"

def get_coordinates_from_place(place_name):
    """Get geographical coordinates for a location name."""
    try:
        geolocator = Nominatim(user_agent="destination_recommender")
        location = geolocator.geocode(place_name)
        if location:
            return location.latitude, location.longitude
        return None, None
    except Exception as e:
        return None, None

def extract_keywords_from_description(text, kw_extractor):
    """Extract keywords from text using YAKE."""
    try:
        keywords = kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]
    except:
        return []

def recommend_by_location_and_keywords(model_info, user_keywords, location_name, top_n=5):
    """Recommend destinations based on keywords and location."""
    if model_info is None:
        return pd.DataFrame(), "Model not available. Please try again later."

    try:
        df = model_info['data']
        vectorizer = model_info['vectorizer']
        tfidf_matrix = model_info['tfidf_matrix']
        kw_extractor = model_info['kw_extractor']
        location_model = model_info['location_model']

        # Get user coordinates
        user_lat, user_lon = get_coordinates_from_place(location_name)
        if user_lat is None or user_lon is None:
            return pd.DataFrame(), "Could not find coordinates for the given location."

        # Get location-based recommendations
        user_coords = np.radians([[user_lat, user_lon]])
        distances, indices = location_model.kneighbors(user_coords)
        
        # Convert distances to kilometers (from radians)
        distances = distances.flatten() * 6371.0  # Earth's radius in km

        # Get keyword-based scores
        user_keywords_expanded = extract_keywords_from_description(user_keywords, kw_extractor)
        user_keywords_text = ' '.join([user_keywords] + user_keywords_expanded)
        user_tfidf = vectorizer.transform([user_keywords_text])
        keyword_scores = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

        # Normalize scores to 0-1 range
        location_scores = 1 / (1 + distances)
        location_scores = location_scores / location_scores.max()
        keyword_scores = (keyword_scores - keyword_scores.min()) / (keyword_scores.max() - keyword_scores.min() + 1e-10)

        # Calculate combined scores for all destinations
        combined_scores = np.zeros(len(df))
        combined_scores[indices[0]] = 0.5 * location_scores + 0.5 * keyword_scores[indices[0]]

        # Get top recommendations
        top_indices = combined_scores.argsort()[::-1][:top_n]
        result_df = df.iloc[top_indices].copy()
        
        # Add distance and relevance information
        result_df['distance_km'] = np.array([
            6371.0 * np.arccos(
                np.clip(
                    np.sin(np.radians(user_lat)) * np.sin(np.radians(lat)) +
                    np.cos(np.radians(user_lat)) * np.cos(np.radians(lat)) *
                    np.cos(np.radians(lon - user_lon)),
                    -1.0, 1.0
                )
            )
            for lat, lon in result_df[['latitude', 'longitude']].values
        ])
        result_df['relevance_score'] = keyword_scores[top_indices] * 100

        return result_df, None

    except Exception as e:
        return pd.DataFrame(), f"Error finding recommendations: {str(e)}"
