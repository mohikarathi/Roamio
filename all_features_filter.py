import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

def train_all_features_model(df):
    """Train the comprehensive recommendation model."""
    try:
        # Create TF-IDF vectorizer for keywords
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

        # Combine cultural description and other text fields if available
        text_data = df['cultural_description'].fillna('')
        if 'description' in df.columns:
            text_data += ' ' + df['description'].fillna('')

        # Fit the vectorizer
        tfidf_matrix = vectorizer.fit_transform(text_data)

        return {
            'vectorizer': vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'data': df,
            'unique_values': {
                'countries': sorted(df['country'].unique()),
                'categories': sorted(df['category'].unique()),
                'religions': sorted(df['majority_religion'].unique()),
                'costs': sorted(df['cost_of_living'].unique()),
                'times': sorted(df['best_time_to_visit'].unique()),
                'safety': sorted(df['safety'].unique())
            }
        }, None
    except Exception as e:
        return None, f"Error training model: {str(e)}"

def recommend_with_all_features(model_info, df, preferences, keywords=''):
    """Get recommendations using all features and keywords."""
    try:
        # Start with all destinations
        filtered_df = df.copy()

        # Apply filters for each preference if not None
        if preferences['country']:
            filtered_df = filtered_df[filtered_df['country'] == preferences['country']]
        if preferences['category']:
            filtered_df = filtered_df[filtered_df['category'] == preferences['category']]
        if preferences['majority_religion']:
            filtered_df = filtered_df[filtered_df['majority_religion'] == preferences['majority_religion']]
        if preferences['cost_of_living']:
            filtered_df = filtered_df[filtered_df['cost_of_living'] == preferences['cost_of_living']]
        if preferences['best_time_to_visit']:
            filtered_df = filtered_df[filtered_df['best_time_to_visit'] == preferences['best_time_to_visit']]
        if preferences['safety']:
            filtered_df = filtered_df[filtered_df['safety'] == preferences['safety']]

        # If no destinations match the criteria, return empty DataFrame
        if filtered_df.empty:
            return pd.DataFrame(), "No destinations match your basic criteria. Try adjusting your selections."

        # If keywords are provided, use NLP to rank results
        if keywords.strip():
            # Transform keywords using the same vectorizer
            keyword_vector = model_info['vectorizer'].transform([keywords])
            
            # Get similarity scores for the filtered destinations only
            filtered_indices = filtered_df.index
            filtered_tfidf = model_info['tfidf_matrix'][filtered_indices]
            
            # Calculate similarity scores for filtered destinations
            sim_scores = cosine_similarity(keyword_vector, filtered_tfidf)
            
            # Add similarity scores to filtered results
            filtered_df['keyword_score'] = sim_scores[0]
            
            # Sort by keyword similarity
            filtered_df = filtered_df.sort_values('keyword_score', ascending=False)
            
            # Keep only destinations with some keyword relevance
            filtered_df = filtered_df[filtered_df['keyword_score'] > 0]

            if filtered_df.empty:
                return pd.DataFrame(), "No destinations match your keyword criteria."

        return filtered_df, None

    except Exception as e:
        return pd.DataFrame(), f"Error finding recommendations: {str(e)}"